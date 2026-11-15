"""動画のペット判定: コマごとに、画像と同じ判定（CLIP + kNN + 撮影日と生存期間）をして、動画1本にまとめる。

 ・参照は、画像の判定と同じ（名前から付いた強いペット名と、人が確認した結果）。動画のコマは、参照に入れない。
 ・動画1本の判定: 判定できたコマのうち最も多かったペット。得票率 = そのペットのコマの割合 × そのコマの得票率の平均。
 ・保存: file_pets に origin='ai', status='hint'（画像のAI判定と同じ）。人が確認した動画（origin='user'）には書かない。
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict

from . import embed
from .identify import level
from .petcheck import allowed_sets, collect_labels, knn_species, species_map


def identify(cfg, conn, k=10, min_class=30, log=print) -> int:
    import numpy as np

    from . import video

    fids, FX = video.load_frames(conn)
    if len(fids) == 0:
        return 0
    ids, X = embed.load_matrix(conn)
    lab = collect_labels(conn)
    cnt = Counter(v[0] for v in lab.values())
    keep = {p for p, c in cnt.items() if c >= min_class}
    ridx = [i for i, a in enumerate(ids) if int(a) in lab and lab[int(a)][0] in keep]
    if len(keep) < 2:
        log(f"（動画のペット判定: 参照にできるペットが2種類に満たないため、行いません: {dict(cnt)}）")
        return 0
    raids = [int(ids[i]) for i in ridx]
    labels = [lab[a][0] for a in raids]
    classes = sorted(keep)
    vaids = sorted({int(a) for a in fids})
    folder = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        folder.setdefault(r["asset_id"], (r["vol_id"], r["volpath"].rsplit("/", 1)[0] if "/" in r["volpath"] else ""))
    allowed_v = dict(zip(vaids, allowed_sets(conn, vaids, classes)))
    qf = [int(a) for a in fids]
    pred, share, ok = knn_species(X[ridx], labels, [lab[a][1] for a in raids], FX, [folder.get(a, ("?", str(a))) for a in qf],
                                  species_map(conn, labels), k=k, allowed=[allowed_v[a] for a in qf])
    by: dict = defaultdict(list)
    for a, p, sh, o in zip(qf, pred, share, ok):
        if o:
            by[a].append((p, float(sh)))
    files_of: dict = defaultdict(list)
    for r in conn.execute("SELECT id, asset_id FROM files WHERE status='present' AND kind='video' AND asset_id IS NOT NULL"):
        files_of[r["asset_id"]].append(r["id"])
    user_files = {r["file_id"] for r in conn.execute("SELECT file_id FROM file_pets WHERE origin='user'")}
    conn.execute("DELETE FROM file_pets WHERE origin='ai' AND file_id IN (SELECT id FROM files WHERE kind='video')")
    stamp = time.strftime("%Y-%m-%d")
    stat: Counter = Counter()
    n_w = 0
    for a in vaids:
        votes = by.get(a)
        if not votes:
            stat[("判定なし", "")] += 1
            continue
        top, n = Counter(p for p, _ in votes).most_common(1)[0]
        sh = (n / len(votes)) * float(np.mean([s for p, s in votes if p == top]))
        lv = level(sh)
        stat[(top, lv)] += 1
        for fid in files_of.get(a, []):
            if fid in user_files:
                continue
            for pet in top.split("+"):
                conn.execute("INSERT OR IGNORE INTO file_pets(file_id, pet_id, status, strength, origin, reason) VALUES(?,?,'hint',?,'ai',?)",
                             (fid, pet, lv, f"video knn {sh:.2f} ({n}/{len(votes)}コマ) {stamp}"))
                n_w += 1
    conn.commit()
    log(f"\n動画のペット判定を保存しました（{len(vaids):,} 本）。ペットごと・自信度ごとの件数:")
    for c in classes:
        log(f"  {c:<8} high {stat[(c, 'high')]:5,} / mid {stat[(c, 'mid')]:5,} / low {stat[(c, 'low')]:5,}")
    if stat[("判定なし", "")]:
        log(f"  判定なし {stat[('判定なし', '')]:,} 本")
    from . import nonlive

    nonlive.run(conn, log=lambda *_: None, cfg=cfg)
    return n_w


def reapply_learned(conn, log=print) -> None:
    """人が「自動で付ける」を押した場所・場面と、見本のある人物について、保存済みのしきい値で自動の判定をやり直す（新しく取り込んだ動画にも付く）。"""
    from . import facelearn, placelearn

    ids, X = embed.load_matrix(conn, with_video=True)
    if len(ids):
        pos = {int(a): i for i, a in enumerate(ids)}
        for kind, (lt, _, col, _) in placelearn.KINDS.items():
            for (pid,) in conn.execute(f"SELECT DISTINCT {col} FROM {lt} WHERE positive=1").fetchall():
                if not placelearn.applied_by_user(conn, pid, kind):       # 人が「自動で付ける」を押したものだけ
                    continue
                n = placelearn.apply(conn, pos, X, ids, pid, placelearn.threshold(conn, pid, kind), kind)
                log(f"  {'場所' if kind == 'place' else '場面'}「{pid}」: 自動で {n:,} 件")
    fi, _, FM = facelearn.load(conn)
    if len(fi) and conn.execute("SELECT 1 FROM face_labels WHERE positive=1 LIMIT 1").fetchone():
        cnt = facelearn.apply_all(conn, fi, FM, lambda p: facelearn.threshold(conn, p))
        for p, n in sorted(cnt.items()):
            log(f"  人物「{p}」: 自動で {n:,} 件の顔")
