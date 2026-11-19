"""検索結果（または選んだ写真・動画）を、別のフォルダに「アルバム」として書き出す。

- 元のファイルは、変更も削除もしない（コピーまたはリンクを作るだけ）。
- 書き出し先は、走査対象のフォルダの中には作れない（再走査で重複扱いになるため）。
- 既存のファイルは上書きしない（同じ名前があれば、番号を付ける）。
- 書き出し先の名前は「撮影日_元のファイル名」（日付順に並ぶ）。一覧は アルバム一覧.csv に残す。
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import time
from pathlib import Path

BAD = '<>:"/\\|?*'
MAX_ITEMS = 20000


def _clean(name: str, default="アルバム") -> str:
    s = "".join("_" if (ch in BAD or ord(ch) < 32) else ch for ch in (name or "")).strip(" .")
    return s[:80] or default


def _inside(child: Path, parent: Path) -> bool:
    try:
        c = os.path.normcase(os.path.realpath(child))
        p = os.path.normcase(os.path.realpath(parent))
        return c == p or c.startswith(p.rstrip("\\/") + os.sep)
    except OSError:
        return False


def normalize_dest(dest: str) -> str:
    """入力の揺れを直す: 円記号（¥）→ \、全角の「：」→ 「:」、ドライブ文字のあとの「:」の抜け（ドライブ文字だけで「:」が無い入力を補う）。"""
    d = (dest or "").strip().strip('"').strip("\u201c\u201d")
    d = d.replace("\u00a5", "\\").replace("\uffe5", "\\").replace("\uff1a", ":").replace("\uff3c", "\\")
    m = re.match(r"^([A-Za-z])[\\/]", d)                          # 「C\アルバム」「C/アルバム」
    if m:
        d = m.group(1) + ":" + d[1:]
    return d


def check_dest(cfg, dest: str, name: str) -> Path:
    """書き出し先フォルダ（dest の下に name のフォルダを作る）を検証して返す（作らない）。"""
    if not dest or not dest.strip():
        raise ValueError("書き出し先のフォルダを入れてください（例: D:\\アルバム）")
    d = Path(normalize_dest(dest))
    if not d.is_absolute():
        raise ValueError("書き出し先は、ドライブ名から始まる完全な場所で入れてください（例: D:\\アルバム）")
    for s in cfg.scans:
        if _inside(d, Path(s["path"])):
            raise ValueError(f"書き出し先が、写真の走査対象フォルダ（{s['path']}）の中です。別の場所を指定してください。")
    if _inside(d, cfg.app_root):
        raise ValueError("書き出し先が、このアプリのフォルダの中です。別の場所を指定してください。")
    parent = d
    while not parent.exists() and parent.parent != parent:
        parent = parent.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError("書き出し先のドライブまたはフォルダが見つかりません。")
    return d / _clean(name)


def _unique_dir(p: Path) -> Path:
    if not p.exists():
        return p
    for i in range(2, 1000):
        q = p.with_name(f"{p.name}_{i}")
        if not q.exists():
            return q
    raise ValueError("同じ名前のフォルダが多すぎます。名前を変えてください。")


def _stem_date(date_taken) -> str:
    s = re.sub(r"\D", "", str(date_taken or ""))[:14]
    return s if len(s) >= 8 else "日付不明"


def collect(conn, ids):
    """asset id → (日付, 実在する元ファイルのパス)。存在しないものは missing に。"""
    from .report import _mounts, abs_path

    mounts = _mounts(conn)
    ids = list(dict.fromkeys(int(i) for i in ids))[:MAX_ITEMS]
    out, missing = [], []
    for k in range(0, len(ids), 500):
        part = ids[k:k + 500]
        qs = ",".join("?" * len(part))
        info = {r["id"]: r["date_taken"] for r in conn.execute(f"SELECT id, date_taken FROM assets WHERE id IN ({qs})", part)}
        paths = {}
        for r in conn.execute(f"SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IN ({qs}) ORDER BY id", part):
            paths.setdefault(r["asset_id"], []).append(abs_path(mounts, r["vol_id"], r["volpath"]))
        for a in part:
            src = next((Path(p) for p in reversed(paths.get(a, [])) if Path(p).is_file()), None)
            (out if src else missing).append((a, info.get(a), src) if src else a)
    # 日付順（日付不明は最後）
    out.sort(key=lambda t: (t[1] is None, str(t[1] or ""), t[0]))
    return out, missing


def export(conn, ids, dest_dir: Path, mode="copy", progress=None, cancel=None) -> dict:
    """ids の元ファイルを dest_dir に書き出す。mode: copy / link（同じドライブなら容量を使わない。できなければコピー）。"""
    items, missing = collect(conn, ids)
    dest_dir.mkdir(parents=True, exist_ok=True)
    used = set()
    rows, done, failed, linked, nbytes = [], 0, [], 0, 0
    for n, (aid, date, src) in enumerate(items, 1):
        if cancel and cancel():
            break
        base = f"{_stem_date(date)}_{_clean(src.name, 'file')}"
        name, i = base, 1
        while name.lower() in used or (dest_dir / name).exists():
            i += 1
            name = f"{Path(base).stem}_{i}{Path(base).suffix}"
        used.add(name.lower())
        tgt = dest_dir / name
        try:
            how = "copy"
            if mode == "link":
                try:
                    os.link(src, tgt)
                    how = "link"
                    linked += 1
                except OSError:
                    shutil.copy2(src, tgt)
            else:
                shutil.copy2(src, tgt)
            nbytes += tgt.stat().st_size
            rows.append((name, str(date or ""), str(src), how))
            done += 1
        except OSError as ex:
            failed.append((str(src), f"{type(ex).__name__}: {ex}"))
        if progress:
            progress(n, len(items), name)
    try:
        with open(dest_dir / "アルバム一覧.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["書き出し名", "撮影日", "元のファイル", "方法"])
            w.writerows(rows)
            for p, m in failed:
                w.writerow(["（失敗）", "", p, m])
    except OSError:
        pass
    return {"done": done, "linked": linked, "failed": failed[:50], "n_failed": len(failed), "missing": len(missing), "bytes": nbytes, "dir": str(dest_dir)}
