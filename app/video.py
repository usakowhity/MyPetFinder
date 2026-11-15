"""動画の取り込み: 動画から数枚のコマを取り出し、コマごとに CLIP の特徴と顔を保存する（途中で止めても続きから再開できる）。

 ・元の動画は読むだけ。結果は video_frames（コマの特徴）、embeddings（動画1本の特徴 = コマの平均）、faces / face_scan（顔）、video_scan に入る。
 ・コマの数は動画の長さで決める（約6秒に1枚、3〜12枚）。等間隔に、各区間の中央から取る。
 ・動画の判定（ペット・場所・場面・人物）は、コマの判定をまとめて行う。コマの特徴を残すので、見本を増やしたあとも、動画を読み直さずに判定し直せる。
 ・顔は同じ動画の中で似たものを1つにまとめる（同じ人が何コマにも写るため）。顔のデータは個人データで、このPCの中だけに置く。
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bench, embed

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mpg", ".mpeg", ".mts", ".m2ts", ".wmv", ".3gp", ".mkv", ".webm", ".flv", ".mod")
SEC_PER_FRAME, MIN_FRAMES, MAX_FRAMES = 6.0, 3, 12
MAXSIDE = 1280
FACE_DEDUP = 0.80          # 同じ動画の中で、これ以上似ている顔は1つにまとめる（SFace のコサイン類似度）


def n_frames_for(duration: float) -> int:
    if not duration or duration <= 0:
        return MIN_FRAMES
    return max(MIN_FRAMES, min(MAX_FRAMES, math.ceil(duration / SEC_PER_FRAME)))


def sample_times(duration: float, n: int) -> list:
    """動画を n 等分し、各区間の中央の時刻（秒）を返す。"""
    if not duration or duration <= 0:
        return [0.0]
    last = max(duration - 0.2, 0.0)
    return [min(duration * (i + 0.5) / n, last) for i in range(n)]


def read_frames(path: str, hint_duration=None, cv2=None):
    """動画から等間隔のコマを取り出す。([(時刻秒, BGR配列), ...], 動画の長さ秒)。読めなければ例外。"""
    import numpy as np

    if cv2 is None:
        import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError("動画を開けません")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        dur = cnt / fps if fps > 1e-3 and cnt > 0 else 0.0
        if dur <= 0 and hint_duration:
            dur = float(hint_duration)
        n = n_frames_for(dur)
        out = []
        times = sample_times(dur, n)
        for t in times:
            ok, img = False, None
            if fps > 1e-3 and cnt > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            h, w = img.shape[:2]
            m = max(h, w)
            if m > MAXSIDE:
                s = MAXSIDE / m
                img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            out.append((float(t), np.ascontiguousarray(img)))
        if not out:
            # 位置の指定が効かない形式: 先頭から順に読む
            cap.release()
            cap = cv2.VideoCapture(str(path))
            ok, img = cap.read()
            if ok and img is not None:
                out.append((0.0, np.ascontiguousarray(img)))
        if not out:
            raise OSError("コマを取り出せません")
        return out, dur
    finally:
        cap.release()


def pending(conn, limit=None, retry=False):
    """まだ取り込んでいない実写の動画。[(asset_id, path, duration)]"""
    from .report import _mounts, abs_path

    mounts = _mounts(conn)
    done = " AND a.id NOT IN (SELECT asset_id FROM video_scan WHERE err IS NULL)" if retry else " AND a.id NOT IN (SELECT asset_id FROM video_scan)"
    rows = conn.execute(
        f"""SELECT a.id aid, a.duration_sec dur, f.vol_id, f.volpath, f.ext, (a.master_file_id = f.id) AS is_master
            FROM assets a JOIN files f ON f.asset_id=a.id AND f.status='present'
            WHERE a.zone='real' AND a.kind='video'{done} ORDER BY a.id""").fetchall()
    best: dict = {}
    for r in rows:
        if "." + str(r["ext"]).lower().lstrip(".") not in VIDEO_EXTS:
            continue
        cur = best.get(r["aid"])
        if cur is None or (r["is_master"] and not cur[1]):
            best[r["aid"]] = (r, r["is_master"])
    out = []
    for aid in sorted(best):
        r = best[aid][0]
        p = abs_path(mounts, r["vol_id"], r["volpath"])
        if os.path.exists(p):
            out.append((aid, p, r["dur"]))
    return out[:limit] if limit else out


def _crop_face(bgr, x, y, w, h, out: Path):
    """顔の切り抜き（検索画面の表示用）を保存する。座標は 0〜1。"""
    import cv2

    H, W = bgr.shape[:2]
    m = 0.35
    x0, y0, x1, y1 = (x - m * w) * W, (y - m * h) * H, (x + (1 + m) * w) * W, (y + (1 + m) * h) * H
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
    a, b, c, d = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)), int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    if c - a < 4 or d - b < 4:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(bgr[b:d, a:c], (112, 112), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run(cfg, conn, limit=None, retry=False, engine=None, with_faces=True, session=None, batch=8, log=print, dino=None, post=True) -> int:
    import numpy as np

    todo = pending(conn, limit, retry)
    done_before = conn.execute("SELECT COUNT(*) FROM video_scan WHERE err IS NULL").fetchone()[0]
    log(f"取り込む動画: {len(todo):,} 本（済み {done_before:,} 本）")
    if not todo:
        return 0
    if session is None:
        import onnxruntime as ort

        try:
            mp = embed.model_path(cfg, "fp32", log)
        except Exception as ex:
            log(f"モデルを取得できませんでした: {ex}")
            return 3
        session = ort.InferenceSession(str(mp), ort.SessionOptions(), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0].name
    outs = [o.name for o in session.get_outputs()]
    if with_faces and engine is None:
        from . import faces as F

        try:
            d = F.model_dir(cfg)
            F.ensure_models(d, log)
            engine = F.Engine(d)
        except Exception as ex:
            log(f"（顔のモデルを使えないため、動画の顔は調べません: {type(ex).__name__}: {ex}）")
            engine = None
    if not with_faces:
        engine = None
    face_dir = Path(cfg.db_path).parent / "thumbs" / "faces"
    stamp = time.strftime("%Y-%m-%d")
    t0 = time.time()
    n_ok = n_bad = n_frames = n_faces = 0
    workers = max(1, (os.cpu_count() or 2) - 2)

    def load(item):
        try:
            return read_frames(item[1], item[2]), None
        except Exception as e:                                   # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"[:200]

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            step = max(workers, 4)
            for s in range(0, len(todo), step):
                part = todo[s: s + step]
                for (aid, _, _), (res, err) in zip(part, ex.map(load, part)):
                    if retry:
                        conn.execute("DELETE FROM video_frames WHERE asset_id=?", (aid,))
                        conn.execute("DELETE FROM faces WHERE asset_id=?", (aid,))
                        conn.execute("DELETE FROM video_pet_frames WHERE asset_id=?", (aid,))
                    if res is None:
                        conn.execute("INSERT OR REPLACE INTO video_scan(asset_id,n_frames,n_faces,err,created) VALUES(?,0,0,?,?)", (aid, err, stamp))
                        n_bad += 1
                        continue
                    frames, dur = res
                    arrs = np.stack([bench.preprocess_bgr(img) for _, img in frames])
                    embs = []
                    for j in range(0, len(arrs), batch):
                        e = bench._pick_embedding(session.run(outs, {inp: arrs[j: j + batch]}))
                        embs.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12))
                    E = np.concatenate(embs).astype(np.float32)
                    conn.execute("DELETE FROM video_frames WHERE asset_id=?", (aid,))
                    conn.executemany("INSERT INTO video_frames(asset_id,idx,t_sec,vec) VALUES(?,?,?,?)",
                                     [(aid, i, frames[i][0], E[i].astype(np.float16).tobytes()) for i in range(len(frames))])
                    if dino is not None:                                  # 公開版: ペット判定用の DINOv2 特徴も、同じコマから作る
                        from PIL import Image

                        from . import dino as D

                        pa = np.stack([D.prep_pil(Image.fromarray(img[:, :, ::-1])) for _, img in frames])
                        P = np.concatenate([dino.encode(pa[j: j + batch]) for j in range(0, len(pa), batch)])
                        conn.execute("DELETE FROM video_pet_frames WHERE asset_id=?", (aid,))
                        conn.executemany("INSERT INTO video_pet_frames(asset_id,idx,vec) VALUES(?,?,?)",
                                         [(aid, i, P[i].astype(np.float16).tobytes()) for i in range(len(P))])
                    mean = E.mean(axis=0)
                    mean = mean / (np.linalg.norm(mean) + 1e-12)
                    conn.execute("INSERT OR REPLACE INTO embeddings(asset_id,model,dim,vec,err,created) VALUES(?,?,?,?,NULL,?)",
                                 (aid, embed.VIDEO_MODEL, int(mean.shape[0]), mean.astype(np.float16).tobytes(), stamp))
                    nf = 0
                    if engine is not None:
                        kept = []
                        for _, img in frames:
                            try:
                                found = engine.detect(img)
                            except Exception:                    # noqa: BLE001
                                continue
                            h, w = img.shape[:2]
                            for (x, y, fw, fh, score, f) in found:
                                if any(float(f @ k) >= FACE_DEDUP for k in kept):
                                    continue
                                kept.append(f)
                                cur = conn.execute("INSERT INTO faces(asset_id,x,y,w,h,score,vec,created) VALUES(?,?,?,?,?,?,?,?)",
                                                   (aid, x / w, y / h, fw / w, fh / h, score, f.astype(np.float16).tobytes(), stamp))
                                try:
                                    _crop_face(img, x / w, y / h, fw / w, fh / h, face_dir / f"{cur.lastrowid}.jpg")
                                except Exception:                # noqa: BLE001
                                    pass
                                nf += 1
                        conn.execute("INSERT OR REPLACE INTO face_scan(asset_id,n,err,created) VALUES(?,?,NULL,?)", (aid, nf, stamp))
                    conn.execute("INSERT OR REPLACE INTO video_scan(asset_id,n_frames,n_faces,err,created) VALUES(?,?,?,NULL,?)", (aid, len(frames), nf, stamp))
                    n_ok += 1
                    n_frames += len(frames)
                    n_faces += nf
                conn.commit()
                el = time.time() - t0
                k = min(s + step, len(todo))
                log(f"  {k:,}/{len(todo):,}  {k / el:5.2f}本/秒  残り約{bench.fmt_time((len(todo) - k) / max(k / el, 1e-9))}  コマ {n_frames:,} / 顔 {n_faces:,} / 失敗 {n_bad}")
    except KeyboardInterrupt:
        conn.commit()
        log("中断しました。もう一度実行すると続きから再開します。")
        return 1
    log(f"完了: 動画 {n_ok:,} 本 / コマ {n_frames:,} 枚 / 顔 {n_faces:,} 件 / 失敗 {n_bad:,} 本 / {bench.fmt_time(time.time() - t0)}")
    if n_ok and post:
        from . import videoai

        videoai.identify(cfg, conn, log=log)
        log("\n見本のある場所・場面・人物の自動判定を、動画にも広げます。")
        videoai.reapply_learned(conn, log=log)
    return 0


def load_frames(conn):
    """(コマごとの asset_id の配列, float32 の行列)。asset_id の昇順・コマの順に並ぶ。"""
    import numpy as np

    rows = conn.execute("SELECT asset_id, vec FROM video_frames ORDER BY asset_id, idx").fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    ids = np.array([r["asset_id"] for r in rows], dtype=np.int64)
    X = np.stack([np.frombuffer(r["vec"], dtype=np.float16) for r in rows]).astype(np.float32)
    return ids, X


_FC: dict = {}


def load_frames_cached(conn):
    """load_frames の結果を、コマの数が変わるまで使い回す（場所・場面の判定を、何度も呼ぶため）。"""
    key = (id(conn), conn.execute("SELECT COUNT(*), COALESCE(MAX(rowid),0) FROM video_frames").fetchone()[:])
    if key[1][0] == 0:
        return load_frames(conn)
    if _FC.get("key") != (key[0], tuple(key[1])):
        _FC.clear()
        _FC["key"] = (key[0], tuple(key[1]))
        _FC["val"] = load_frames(conn)
    return _FC["val"]
