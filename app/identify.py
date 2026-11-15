"""Step 2 本番: 実写の全画像に、AI（CLIP の特徴ベクトル + kNN + 撮影日と生存期間）のペット判定を付けて DB に保存する。

  ・参照データ: 名前から付いた強いペット名と、人が確認した結果（collect_labels と同じ）
  ・判定: 同じフォルダ以外の似た画像に投票。撮影日が信頼できるときは、期間外のペットを候補から外す
  ・保存: file_pets に origin='ai', status='hint'。strength は得票率で high(0.9〜) / mid(0.7〜) / low。reason に得票率
  ・人が確認した画像（origin='user'）には書かない。AIの結果は、実行のたびに入れ替える（名前の推定や人の確認は触らない）
"""
from __future__ import annotations

import time
from collections import Counter

from . import embed
from .petcheck import allowed_sets, collect_labels, knn_species, species_map

HIGH, MID = 0.9, 0.7
# ネズミと一緒にいた時代（Momo）の写真は、ネズミらしさで決める。確認用80枚の結果: 0.1以上はほぼ Momo（93%）、0.3以上は95%
RAT_MIN, RAT_HIGH = 0.1, 0.3


def level(share: float) -> str:
    return "high" if share >= HIGH else "mid" if share >= MID else "low"


def run(cfg, conn, k=10, min_class=30, model=None, log=print) -> int:
    import numpy as np

    model = model or embed.pet_model(conn)
    log(f"ペット判定に使う特徴: {'DINOv2' if model == embed.DINO_MODEL else 'CLIP'}")
    ids, X = embed.load_matrix(conn, model=model)
    if len(ids) == 0:
        log("特徴ベクトルがありません。先にメニュー8（Embed）を実行してください。")
        return 2
    lab = collect_labels(conn)
    cnt = Counter(v[0] for v in lab.values())
    keep = {p for p, c in cnt.items() if c >= min_class}
    ridx = [i for i, a in enumerate(ids) if int(a) in lab and lab[int(a)][0] in keep]
    if len(keep) < 2:
        log(f"参照にできるペットが2種類に満たない（{dict(cnt)}）ので判定できません。")
        return 3
    # 判定の対象: 実写の画像アセットすべて
    real = {r["id"] for r in conn.execute("SELECT id FROM assets WHERE zone='real' AND kind='image'")}
    qidx = [i for i, a in enumerate(ids) if int(a) in real]
    folder = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        folder.setdefault(r["asset_id"], (r["vol_id"], r["volpath"].rsplit("/", 1)[0] if "/" in r["volpath"] else ""))
    raids = [int(ids[i]) for i in ridx]
    qaids = [int(ids[i]) for i in qidx]
    labels = [lab[a][0] for a in raids]
    classes = sorted(keep)
    log(f"参照 {len(raids):,} 件 / 判定する画像 {len(qaids):,} 件 / ペット {', '.join(classes)}")
    allowed = allowed_sets(conn, qaids, classes)
    species_of = species_map(conn, labels)
    pred, share, ok = knn_species(X[ridx], labels, [lab[a][1] for a in raids], X[qidx],
                                  [folder.get(a, ("?", str(a))) for a in qaids], species_of, k=k, allowed=allowed)

    # ネズミと一緒にいた時代は、写真の見た目（ウサギらしい／ネズミらしい）で、Momo か Usako かを決める
    rs = {r["asset_id"]: r["rat_score"] for r in conn.execute("SELECT asset_id, rat_score FROM asset_presence")}
    override: dict[int, tuple] = {}
    for i, a in enumerate(qaids):
        al = allowed[i]
        if al is None or a not in rs:
            continue
        rats = [c for c in al if species_of.get(c) == "rat"]
        rabbits = [c for c in al if species_of.get(c) == "rabbit"]
        if len(rats) == 1 and rabbits and rs[a] >= RAT_MIN:
            override[i] = (rats[0], "high" if rs[a] >= RAT_HIGH else "mid", f"ratscore {rs[a]:.2f}")
    if override:
        log(f"ネズミらしさで決めた写真（Momoの時代）: {len(override):,} 件")
    elif not rs:
        log("（ネズミらしさが未計算です。メニュー12 を実行すると、Momo の判定が加わります）")

    files_of: dict[int, list[int]] = {}
    for r in conn.execute("SELECT id, asset_id FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        files_of.setdefault(r["asset_id"], []).append(r["id"])
    user_files = {r["file_id"] for r in conn.execute("SELECT file_id FROM file_pets WHERE origin='user'")}
    conn.execute("DELETE FROM file_pets WHERE origin='ai' AND file_id NOT IN (SELECT id FROM files WHERE kind='video')")
    stamp = time.strftime("%Y-%m-%d")
    n_w = 0
    stat: Counter = Counter()
    for i, (a, p, sh, o) in enumerate(zip(qaids, pred, share, ok)):
        reason = None
        if i in override:
            p, lv, reason = override[i]
            sh, o = 1.0, True
        if not o:
            stat[("判定なし", "")] += 1
            continue
        if reason is None:
            lv = level(float(sh))
            reason = f"knn {float(sh):.2f} {stamp}"
        stat[(p, lv)] += 1
        for fid in files_of.get(a, []):
            if fid in user_files:
                continue
            for pet in p.split("+"):          # 「Momo+Usako」は、2匹分の行にする
                conn.execute(
                    """INSERT OR IGNORE INTO file_pets(file_id, pet_id, status, strength, origin, reason)
                       VALUES(?,?,'hint',?,'ai',?)""", (fid, pet, lv, reason))
                n_w += 1
    conn.commit()
    log(f"\nAIの判定を保存しました（{n_w:,} ファイル分）。ペットごと・自信度ごとの件数（アセット単位）:")
    for c in classes:
        log(f"  {c:<8} high {stat[(c, 'high')]:6,} / mid {stat[(c, 'mid')]:5,} / low {stat[(c, 'low')]:5,}")
    if stat[("判定なし", "")]:
        log(f"  判定なし {stat[('判定なし', '')]:,} 件")
    # 名前の推定との食い違い（確認の目安）
    bad = conn.execute(
        """SELECT COUNT(DISTINCT a.file_id) FROM file_pets a
           WHERE a.origin='ai' AND a.strength='high'
             AND EXISTS (SELECT 1 FROM file_pets n WHERE n.file_id=a.file_id AND n.origin='name_hint' AND n.strength='strong')
             AND NOT EXISTS (SELECT 1 FROM file_pets n WHERE n.file_id=a.file_id AND n.pet_id=a.pet_id
                             AND n.origin IN ('name_hint','user') AND n.status!='rejected')"""
    ).fetchone()[0]
    log(f"\n名前の推定（強）と食い違い、AIの自信が高い（high）ファイル: {bad:,} 件。")
    from . import nonlive, videoai
    videoai.identify(cfg, conn, k=k, min_class=min_class, log=log)        # 取り込み済みの動画があれば、動画も判定し直す
    nonlive.run(conn, log=log, cfg=cfg)
    return 0
