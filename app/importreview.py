"""review_ai_folders.csv の「正しいペット」欄を DB に取り込む。

  ・書かれたペットは、そのフォルダ直下のファイルに「人が確認した」(origin=user, status=identified) として付く。
    フォルダ名の推定（name_hint）は消さず、再スキャンでも上書きされない。
  ・書かれていないペットで、名前の推定に出ていたものは「写っていない」(rejected) にする。
  ・'-' は「ペットを特定しない」。file_pets は変えず、folder_reviews に記録して、AI検証の正解データから外す。
  ・空欄は何もしない。サブフォルダには及ばない（CSVの行はフォルダ単位）。
"""
from __future__ import annotations

import re
import time
from collections import defaultdict

from .report import _mounts, abs_path
from .reviewio import read_table
from .textnorm import norm

NONE_MARKS = {"-", "ー", "－", "―", "−", "なし", "無し", "不明"}
SPLIT = re.compile(r"[,、，/／・+＋＆&\s]+")


def dir_key(path: str) -> str:
    return re.sub(r"[\\/]+", "/", path.strip()).rstrip("/").lower()


def _pet_lookup(conn):
    ids = {r["id"].lower(): r["id"] for r in conn.execute("SELECT id FROM pets")}
    al = {r["alias_norm"]: r["pet_id"] for r in conn.execute("SELECT alias_norm, pet_id FROM pet_aliases")}

    def look(tok):
        t = tok.strip()
        return ids.get(t.lower()) or al.get(norm(t))
    return look


def run(conn, path: str, log=print) -> int:
    try:
        rows = read_table(path)
    except Exception as ex:
        log(f"ファイルを読めませんでした: {ex}")
        return 2
    if not rows or "フォルダ" not in rows[0] or "正しいペット" not in rows[0]:
        log("「フォルダ」「正しいペット」の列が見つかりません。review_ai_folders.csv を使ってください。")
        return 2
    look = _pet_lookup(conn)
    mounts = _mounts(conn)
    wanted = {dir_key(r["フォルダ"]): r for r in rows if r["正しいペット"].strip()}
    by_dir: dict[str, list[int]] = defaultdict(list)
    for f in conn.execute("SELECT id, vol_id, volpath FROM files WHERE status='present'"):
        full = abs_path(mounts, f["vol_id"], f["volpath"])
        d = dir_key(full.rsplit("\\", 1)[0] if "\\" in full else full.rsplit("/", 1)[0])
        if d in wanted:
            by_dir[d].append(f["id"])
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_set = n_rej = n_none = n_files = 0
    for key, r in wanted.items():
        ans = r["正しいペット"].strip()
        conn.execute("INSERT OR REPLACE INTO folder_reviews(folder_key, folder, answer, created) VALUES(?,?,?,?)",
                     (key, r["フォルダ"], ans, stamp))
        if ans in NONE_MARKS:
            n_none += 1
            continue
        pets, bad = [], []
        for tok in filter(None, SPLIT.split(ans)):
            p = look(tok)
            (pets if p else bad).append(p or tok)
        if bad or not pets:
            log(f"  注意: 登録にないペット名 {bad} → このフォルダは取り込みません: {r['フォルダ']}")
            conn.execute("DELETE FROM folder_reviews WHERE folder_key=?", (key,))
            continue
        files = by_dir.get(key, [])
        if not files:
            log(f"  注意: 目録にこのフォルダのファイルがありません: {r['フォルダ']}")
        for fid in files:
            for p in pets:
                conn.execute(
                    """INSERT OR REPLACE INTO file_pets(file_id, pet_id, status, strength, origin, reason)
                       VALUES(?,?,'identified',NULL,'user','folder review')""", (fid, p))
            for old in conn.execute("SELECT pet_id FROM file_pets WHERE file_id=? AND origin='name_hint' AND status='hint'", (fid,)).fetchall():
                if old["pet_id"] not in pets:
                    conn.execute("UPDATE file_pets SET status='rejected', origin='user', reason='folder review' WHERE file_id=? AND pet_id=?",
                                 (fid, old["pet_id"]))
                    n_rej += 1
        n_set += 1
        n_files += len(files)
    conn.commit()
    log(f"取り込み完了: ペットを確定したフォルダ {n_set} 件（ファイル {n_files:,} 件） / 「特定しない(-)」{n_none} 件 / 「写っていない」にした推定 {n_rej:,} 件")
    return 0
