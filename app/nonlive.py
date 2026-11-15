"""生きた登録ペットがいるはずのない写真（全ペットの亡くなった後・迎える前）に印を付ける。

撮影日（日付が撮影由来で、日まで分かるものだけ）で、登録ペット全員が「生存期間外（weak）」になる写真は、
Usako・Kuro などの名前が写っていても生きた姿ではない（遺影・置物・画面・他の動物・人物のみ）。
名前のタグは消さず、asset_nonlive に別のフラグとして記録する。
 kind: 'memorial_candidate' … NoPet の点数が高い（遺影・置物・画面などの候補） / 'other' … それ以外（他の動物・人物など、要確認）
 ユーザーが「ペット名」と確認した写真（photo_reviews）は対象外。
"""
from __future__ import annotations

import time
from collections import Counter

NOPET_MIN = 0.5


def run(conn, log=print, cfg=None) -> int:
    from . import nopetfolders
    from .pethints import PetIndex

    pi = PetIndex(conn)
    pets = [r["id"] for r in conn.execute("SELECT id FROM pets")]
    nopet = {r["asset_id"]: r["score"] for r in conn.execute("SELECT asset_id, score FROM asset_nopet")}
    confirmed = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM photo_reviews WHERE answer<>'none' AND asset_id IS NOT NULL")}
    named = {r["asset_id"] for r in conn.execute(
        """SELECT DISTINCT f.asset_id FROM file_pets p JOIN files f ON f.id=p.file_id
           WHERE p.status!='rejected' AND f.asset_id IS NOT NULL""")}
    stamp = time.strftime("%Y-%m-%d")
    rows, cnt = [], Counter()
    for a in conn.execute("SELECT id, date_taken, date_confidence, date_source, date_conflict FROM assets WHERE zone='real' AND kind IN ('image','video') AND date_taken IS NOT NULL"):
        if a["id"] in confirmed:
            continue
        if pets and all(pi.strength(p, a["date_taken"], a["date_confidence"], a["date_source"], bool(a["date_conflict"]))[0] == "weak" for p in pets):
            s = nopet.get(a["id"])
            kind = "memorial_candidate" if s is not None and s >= NOPET_MIN else "other"
            rows.append((a["id"], kind, "no registered pet alive on the shooting date", s, stamp))
            cnt[(kind, a["id"] in named)] += 1
    # 登録フォルダ（config/nopet_folders.toml の live=false）の写真は、日付に関係なく印を付ける
    n_folder = 0
    if cfg is not None:
        fo = [f for f in nopetfolders.load(cfg) if not f["live"]]
        by_kind, _ = nopetfolders.assets_by_kind(conn, fo)
        have = {r[0] for r in rows}
        for (kind, _live), ids in by_kind.items():
            for a in ids:
                if a in confirmed:
                    continue
                if a in have:
                    rows = [r for r in rows if r[0] != a]
                else:
                    n_folder += 1
                rows.append((a, "memorial_candidate", f"registered folder: {kind}", nopet.get(a), stamp))
        have = None
    conn.execute("DELETE FROM asset_nonlive")
    conn.executemany("INSERT INTO asset_nonlive(asset_id, kind, reason, nopet, created) VALUES(?,?,?,?,?)", rows)
    conn.commit()
    log(f"\n生きたペットがいない写真: {len(rows):,} 件（名前のタグは残し、別のフラグを付けました）")
    if cfg is not None:
        log(f"  うち、登録フォルダ（遺影・置物など）の写真で、撮影日では分からなかったもの: {n_folder:,} 件")
    log(f"  遺影・置物・画面などの候補（NoPet {NOPET_MIN}以上）: {sum(1 for r in rows if r[1] == 'memorial_candidate'):,} 件")
    log(f"  それ以外（他の動物・人物のみ・要確認）: {sum(1 for r in rows if r[1] == 'other'):,} 件")
    return len(rows)
