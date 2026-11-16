"""公開版（MyPetFinder）の中核。個人用の設定・フォルダ名の手掛かりに頼らず、次の流れだけで動く。

  1. 初回: 写真・動画のあるフォルダを選ぶ（config/settings.toml を自動で作る）
  2. 解析: 目録作成 → CLIP（場所・場面・文章検索）→ DINOv2（ペットの個体）を、ワンボタンで順に実行
  3. 名前付け: 画面でグループに名前を付ける（petlearn）。付けた名前が見本になる
  4. 判定: 見本に似ているかだけで、残りの写真に名前を付ける（simple_identify。撮影日・生存期間は使わない）

対象: 種類の違うペット（ウサギ・犬・猫・ネズミ…）の組み合わせ、または1匹だけ。
対象外: 同じ種類で見た目がそっくりな個体どうし（人でも見た目では区別できない）。その場合は警告を出す。
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

from . import db, embed, petlearn

MIN_EXAMPLES = 5            # 1匹あたり、見本がこれ未満なら判定に使わない
K = 3                       # 似ている見本の上位 K 枚の平均で測る
LOOKALIKE = 0.15            # 同じ種類の2匹で、これ以上が取り違えられたら「見た目が似ている」と警告
LEVELS = (("high", 0.25), ("mid", 0.10), ("low", 0.03))     # 見本どうしの類似度の分位点（下位から）

DEFAULT_EXTS = {
    "image_exts": [".jpg", ".jpeg", ".jpe", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"],
    "video_exts": [".avi", ".mp4", ".mov", ".m4v", ".3gp", ".mpg", ".mpeg", ".wmv", ".mts", ".m2ts", ".mkv", ".flv", ".webm", ".asf"],
    "exclude": ["$RECYCLE.BIN", "System Volume Information", "Recovery", "Windows", "Program Files", "Program Files (x86)",
                "ProgramData", "AppData", ".git", "node_modules", "__pycache__"],
}


# ---------------------------------------------------------------- 設定
def settings_path(app_root: Path) -> Path:
    return Path(app_root) / "config" / "settings.toml"


def check_folders(folders) -> list[str]:
    out = []
    for f in folders:
        f = str(f).strip().strip('"')
        if not f:
            continue
        if "'" in f or "\n" in f or "\r" in f:
            raise ValueError(f"使えない文字が含まれています: {f}")
        p = Path(f)
        if not p.is_absolute():
            raise ValueError(f"フルパスで指定してください（例: D:\\写真）: {f}")
        if p.parent == p:
            raise ValueError(f"ドライブ全体（{f}）は指定できません。時間がかかり、ペットと無関係なファイルも含まれます。写真・動画のあるフォルダを指定してください（例: D:\\Photos\\Pet）: {f}")
        if not p.is_dir():
            raise ValueError(f"フォルダが見つかりません: {f}")
        out.append(str(p))
    if not out:
        raise ValueError("フォルダを1つ以上指定してください。")
    return list(dict.fromkeys(out))


def write_settings(app_root: Path, folders) -> Path:
    folders = check_folders(folders)
    path = settings_path(app_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = lambda xs: "[" + ", ".join('"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"' for x in xs) + "]"
    lines = ["# MyPetFinder の設定（初回の画面で自動作成。あとから、フォルダを足したいときは下の [[scan]] を増やします）",
             "[general]", 'db_path = "data/mypet.db"', 'hash_mode = "quick"',
             f"image_exts = {q(DEFAULT_EXTS['image_exts'])}", f"video_exts = {q(DEFAULT_EXTS['video_exts'])}",
             f"exclude_dir_names = {q(DEFAULT_EXTS['exclude'])}", "ai_path_keywords = []", ""]
    for f in folders:
        lines += ["[[scan]]", f"path = '{f}'", 'zone = "real"', ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    pets = path.parent / "pets.toml"
    if not pets.exists():
        pets.write_text("# ペット台帳（画面の「＋ 新しいペット」で追記されます）\n", encoding="utf-8")
    return path


def suggest_folders() -> list[str]:
    """よくある保存先の候補（存在するものだけ）。"""
    home = Path(os.path.expanduser("~"))
    cands = [home / n for n in ("Pictures", "Videos", "Documents", "Desktop", "OneDrive/Pictures", "OneDrive/画像", "画像", "ピクチャ", "ビデオ")]
    return [str(c) for c in cands if c.is_dir()]


# ---------------------------------------------------------------- 判定
def _class_scores(X, ids, refs_by_pet, pos):
    """各画像 × 各ペットの類似度（見本の上位 K 枚の平均。自分自身は除く）。(pets, scores[N, C])"""
    import numpy as np

    pets = sorted(refs_by_pet)
    S = np.zeros((len(ids), len(pets)), dtype=np.float32)
    for j, p in enumerate(pets):
        rows = np.array([pos[a] for a in refs_by_pet[p]], dtype=int)
        selfrows = np.full(len(ids), -1, dtype=int)
        selfrows[rows] = np.arange(len(rows))
        S[:, j] = petlearn.topk_mean(X, X[rows], K, self_rows=selfrows)
    return pets, S


def simple_identify(cfg, conn, log=print, min_examples=MIN_EXAMPLES) -> dict:
    """人が付けた名前（と、名前から分かるもの）を見本に、残りの写真へ名前を付ける。"""
    import numpy as np

    ids, X = petlearn.load(conn)
    if len(ids) == 0:
        log("特徴ベクトルがありません。先に解析を実行してください。")
        return {"error": "no_embeddings"}
    model = embed.pet_model(conn)
    pos = {int(a): i for i, a in enumerate(ids)}
    lab = petlearn.labelled(conn)
    by: dict[str, list[int]] = {}
    for a, p in lab.items():
        if a in pos:
            by.setdefault(p, []).append(a)
    keep = {p: v for p, v in by.items() if len(v) >= min_examples}
    short = {p: len(v) for p, v in by.items() if len(v) < min_examples}
    if short:
        log(f"見本が{min_examples}枚に満たないペットは、まだ判定に使いません: {short}")
    if not keep:
        log("見本にできるペットがありません。「ペットに名前を付ける」で、グループに名前を付けてください。")
        return {"error": "no_examples", "short": short}
    log(f"判定に使う特徴: {'DINOv2' if model == embed.DINO_MODEL else 'CLIP'} / 見本 {sum(map(len, keep.values())):,} 枚 / ペット {len(keep)} 匹")
    pets, S = _class_scores(X, ids, keep, pos)
    # 各ペットの「見本どうしの類似度」の分布から、自分の基準を決める（ペットごとに写りやすさが違うため）
    thr = {}
    for j, p in enumerate(pets):
        own = S[[pos[a] for a in keep[p]], j]
        thr[p] = {lv: float(np.quantile(own, q)) for lv, q in LEVELS}
    best = S.argmax(axis=1)
    sc = S[np.arange(len(ids)), best]
    second = np.partition(S, -2, axis=1)[:, -2] if S.shape[1] > 1 else np.full(len(ids), -1.0)
    skip_assets = petlearn._ignored(conn)
    user_files = {r[0] for r in conn.execute("SELECT file_id FROM file_pets WHERE origin='user'")}
    files_of: dict[int, list[int]] = {}
    for r in conn.execute("SELECT id, asset_id FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        files_of.setdefault(r["asset_id"], []).append(r["id"])
    conn.execute("DELETE FROM file_pets WHERE origin='ai'")
    stat: Counter = Counter()
    n_w = 0
    stamp = time.strftime("%Y-%m-%d")
    for i, a in enumerate(ids):
        a = int(a)
        if a in skip_assets:
            continue
        p = pets[best[i]]
        lv = next((l for l, _ in LEVELS if sc[i] >= thr[p][l]), None)
        if lv == "high" and len(pets) > 1 and sc[i] - second[i] < 0.03:
            lv = "mid"                                   # 2匹のどちらにも似ている → 自信は下げる
        if lv is None:
            stat[("判定なし", "")] += 1
            continue
        stat[(p, lv)] += 1
        for fid in files_of.get(a, []):
            if fid in user_files:
                continue
            conn.execute("""INSERT OR IGNORE INTO file_pets(file_id,pet_id,status,strength,origin,reason)
                            VALUES(?,?,'hint',?,'ai',?)""", (fid, p, lv, f"simple {sc[i]:.2f} {stamp}"))
            n_w += 1
    conn.commit()
    names = {r["id"]: r["display_name"] or r["id"] for r in conn.execute("SELECT id, display_name FROM pets")}
    for p in pets:
        log(f"  {names.get(p, p):<10} high {stat[(p, 'high')]:6,} / mid {stat[(p, 'mid')]:5,} / low {stat[(p, 'low')]:5,}")
    log(f"  判定なし {stat[('判定なし', '')]:,} 枚（ペットが写っていない写真など）")
    vstat = identify_videos(conn, pets, keep, thr, S, pos, X, files_of, user_files, stamp, log)
    warn = lookalikes(conn, pets, S, keep, pos, log=log)
    save_warnings(conn, warn)
    return {"pets": pets, "stat": {f"{p}:{l}": c for (p, l), c in stat.items()}, "warnings": warn, "videos": vstat}


def identify_videos(conn, pets, keep, thr, S, pos, X, files_of, user_files, stamp, log) -> dict:
    """動画: コマごとに写真と同じ方法（DINOv2）で判定し、ペットが写っているコマの割合で動画の判定にする。
    ペットが複数のコマに写っていれば high、1/3 以上のコマなら mid、1コマだけなら low。"""
    import numpy as np

    rows = conn.execute("SELECT asset_id, vec FROM video_pet_frames ORDER BY asset_id, idx").fetchall()
    if not rows:
        return {}
    aid = np.array([r["asset_id"] for r in rows], dtype=np.int64)
    V = np.stack([np.frombuffer(r["vec"], dtype=np.float16) for r in rows]).astype(np.float32)
    F = np.zeros((len(V), len(pets)), dtype=np.float32)
    for j, p in enumerate(pets):
        F[:, j] = petlearn.topk_mean(V, X[[pos[a] for a in keep[p]]], K)
    hit = np.stack([F[:, j] >= thr[p]["mid"] for j, p in enumerate(pets)], axis=1)       # そのコマに、そのペットが写っている
    if len(pets) > 1:
        top = F.argmax(axis=1)
        hit &= (np.arange(len(pets))[None, :] == top[:, None])                          # 1コマ1匹（似たほうだけ）
    ignored = petlearn._ignored(conn)
    out: Counter = Counter()
    order = np.argsort(aid, kind="stable")
    aid, hit = aid[order], hit[order]
    bounds = np.flatnonzero(np.r_[True, aid[1:] != aid[:-1], True])
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        a = int(aid[b0])
        if a in ignored:
            continue
        n = b1 - b0
        for j, p in enumerate(pets):
            h = int(hit[b0:b1, j].sum())
            if h == 0:
                continue
            lv = "high" if h >= 2 and h / n >= 0.5 else "mid" if h / n >= 1 / 3 else "low"
            out[(p, lv)] += 1
            for fid in files_of.get(a, []):
                if fid not in user_files:
                    conn.execute("INSERT OR IGNORE INTO file_pets(file_id,pet_id,status,strength,origin,reason) VALUES(?,?,'hint',?,'ai',?)",
                                 (fid, p, lv, f"video {h}/{n} {stamp}"))
    conn.commit()
    names = {r["id"]: r["display_name"] or r["id"] for r in conn.execute("SELECT id, display_name FROM pets")}
    log(f"動画 {len(bounds) - 1:,} 本を判定:")
    for p in pets:
        log(f"  {names.get(p, p):<10} high {out[(p, 'high')]:5,} / mid {out[(p, 'mid')]:5,} / low {out[(p, 'low')]:5,}")
    return {f"{p}:{l}": c for (p, l), c in out.items()}


def lookalikes(conn, pets, S, keep, pos, log=print) -> list[dict]:
    """同じ種類の2匹で、見本どうしが取り違えられやすい組を探す。"""
    import numpy as np

    sp = {r["id"]: (r["species"] or "").strip().lower() for r in conn.execute("SELECT id, species FROM pets")}
    names = {r["id"]: r["display_name"] or r["id"] for r in conn.execute("SELECT id, display_name FROM pets")}
    out = []
    for ja, a in enumerate(pets):
        for jb, b in enumerate(pets):
            if a == b or (sp.get(a) and sp.get(b) and sp[a] != sp[b]):
                continue                                   # 種類が違えば取り違えない
            rows = [pos[x] for x in keep[a]]
            rate = float((S[rows, jb] > S[rows, ja]).mean())
            if rate >= LOOKALIKE:
                out.append({"a": a, "b": b, "a_name": names.get(a, a), "b_name": names.get(b, b), "rate": round(rate, 2)})
    for w in out:
        log(f"注意: 「{w['a_name']}」の見本の {w['rate'] * 100:.0f}% が「{w['b_name']}」と取り違えられました。見た目が似ていて、写真だけでは区別できません。")
    return out


def save_warnings(conn, warn) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('lookalikes', ?)", (json.dumps(warn, ensure_ascii=False),))
    conn.commit()


def load_warnings(conn) -> list[dict]:
    r = conn.execute("SELECT value FROM meta WHERE key='lookalikes'").fetchone()
    try:
        return json.loads(r[0]) if r else []
    except (ValueError, TypeError):
        return []


# ---------------------------------------------------------------- 解析（ワンボタン）
STAGES = ("目録作成", "場所・場面の特徴（CLIP）", "ペット個体の特徴（DINOv2）", "ペットの判定", "動画のコマ解析")


def status(conn) -> dict:
    def n(q, *a):
        return conn.execute(q, a).fetchone()[0]
    return {"files": n("SELECT COUNT(*) FROM files WHERE status='present'"),
            "images": n("SELECT COUNT(*) FROM assets WHERE kind='image' AND zone='real'"),
            "clip": n("SELECT COUNT(*) FROM embeddings WHERE model=? AND vec IS NOT NULL", embed.MODEL_NAME),
            "dino": n("SELECT COUNT(*) FROM embeddings WHERE model=? AND vec IS NOT NULL", embed.DINO_MODEL),
            "videos": n("SELECT COUNT(*) FROM assets WHERE kind='video' AND zone='real'"),
            "videos_done": n("SELECT COUNT(DISTINCT asset_id) FROM video_pet_frames"),
            "named": n("SELECT COUNT(DISTINCT f.asset_id) FROM file_pets fp JOIN files f ON f.id=fp.file_id "
                       "WHERE fp.origin='user' AND fp.status='identified'")}


class Job:
    """バックグラウンドの解析。1つだけ動く。"""

    def __init__(self):
        self.running = False
        self.stage = ""
        self.log: list[str] = []
        self.error = None
        self.done = False

    def snapshot(self):
        return {"running": self.running, "stage": self.stage, "error": self.error, "done": self.done, "log": self.log[-6:]}

    def start(self, cfg, stages=(0, 1, 2, 3)):
        if self.running:
            raise ValueError("解析中です。終わるまでお待ちください。")
        self.running, self.error, self.done, self.log = True, None, False, []

        def say(m):
            self.log.append(str(m))

        def run():
            conn = db.connect(cfg, check_same_thread=False)
            try:
                db.sync_registry(conn, cfg)
                from . import dino, scanner
                if 0 in stages:
                    self.stage = STAGES[0]
                    scanner.run_scan(cfg, conn, log=say)
                if 1 in stages:
                    self.stage = STAGES[1]
                    embed.run(cfg, conn, log=say)
                    from . import zeroshot

                    say("文章検索用のモデルを確認します…")
                    zeroshot.ensure_files(cfg.app_root / "models" / embed.MODEL_NAME, say)
                if 2 in stages:
                    self.stage = STAGES[2]
                    dino.run(cfg, conn, log=say)
                if 3 in stages:
                    self.stage = STAGES[4]
                    from . import dino as D, video

                    enc = D.Encoder(D.model_path(cfg, say))
                    video.run(cfg, conn, with_faces=False, dino=enc, post=False, log=say)
                self.stage = STAGES[3]
                if conn.execute("SELECT 1 FROM file_pets WHERE origin IN ('user','name_hint') LIMIT 1").fetchone():
                    simple_identify(cfg, conn, log=say)
                self.done = True
            except Exception as ex:                           # noqa: BLE001
                self.error = f"{type(ex).__name__}: {ex}"
            finally:
                self.running = False
                conn.close()
        threading.Thread(target=run, daemon=True).start()


def identify_job(cfg, job: Job):
    """名前付けのあとの「判定をやり直す」だけ。"""
    if job.running:
        raise ValueError("処理中です。終わるまでお待ちください。")
    job.running, job.error, job.done, job.log, job.stage = True, None, False, [], STAGES[3]

    def run():
        conn = db.connect(cfg, check_same_thread=False)
        try:
            simple_identify(cfg, conn, log=lambda m: job.log.append(str(m)))
            job.done = True
        except Exception as ex:                               # noqa: BLE001
            job.error = f"{type(ex).__name__}: {ex}"
        finally:
            job.running = False
            conn.close()
    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- 起動（初回の設定画面 → 検索画面）
def serve_setup(app_root: Path, port=8765, open_browser=True, log=print) -> bool:
    """設定がまだ無いとき、フォルダを選ぶ画面を出す。保存できたら True。"""
    import re
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    page = (Path(__file__).parent / "web" / "setup.html").read_bytes()
    state = {"ok": False}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ct):
            self.send_response(code)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self):
            return (self.headers.get("Host") or "").split(":")[0].lower() in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            if self.path == "/api/setup/suggest":
                return self._json({"folders": suggest_folders()})
            if self.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self._host_ok() or (origin and origin.split("://", 1)[-1].split(":")[0].lower() not in ("127.0.0.1", "localhost")):
                return self._send(403, b"forbidden", "text/plain")
            if self.path != "/api/setup/save" or "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 1 << 16)).decode("utf-8") or "{}")
                write_settings(app_root, body.get("folders", []))
                state["ok"] = True
                self._json({"ok": True})
            except ValueError as ex:
                self._json({"error": str(ex)}, 400)

    srv = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), H)
            break
        except OSError:
            continue
    if srv is None:
        log("使える番号（ポート）が見つかりません。")
        return False
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    log(f"はじめの設定: {url}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    srv.handle_error = lambda request, client_address: None
    srv.timeout = 0.5
    try:
        while not state["ok"]:
            srv.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return state["ok"]


def main(app_root: Path | None = None, port=8765, open_browser=True) -> int:
    from .config import ConfigError, load_settings

    app_root = Path(app_root) if app_root else Path(__file__).resolve().parent.parent
    if not settings_path(app_root).exists():
        if not serve_setup(app_root, port, open_browser):
            return 1
    try:
        cfg = load_settings(settings_path(app_root), app_root=app_root)
    except ConfigError as ex:
        print(f"設定エラー: {ex}")
        return 2
    conn = db.connect(cfg)
    db.sync_registry(conn, cfg)
    conn.close()
    from . import webui

    return webui.serve(cfg, port=port, open_browser=open_browser, public=True)
