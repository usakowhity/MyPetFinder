"""「ペットが写っていない写真」らしさ（nopet score）。

人が「どちらもいない」と確認した写真（photo_reviews）を負例、名前から付いた強いペット名の写真を正例にして学習した線形モデル。
フォルダ名が Kuro でも、人・景色・置物・縫いぐるみ・遺影だけの写真には、ペットの名前を付けたくない、という用途。
 ・フォルダ登録（nopet_folders.toml）の「他のウサギ」は、生きた動物が写っている側の学習データに使う
 ・学習に使っていないフォルダの写真での当たり方（検出率と誤検出率）を表示する
 ・全画像に点数を付けて asset_nopet に保存する。Identify にはまだ使わない（精度を確認してから）
 ・確認用に、点数の層ごとに写真を抜き出して review_nopet.csv に出す（「正しいペット」欄に書くと import-photos で取り込める）
"""
from __future__ import annotations

import random
import time
from collections import defaultdict

from . import embed, nopetfolders, zeroshot
from .petcheck import collect_labels
from .presence import fit_logistic, score
from .report import _mounts, abs_path, _write_csv


def run(cfg, conn, log=print, per_bin=20, seed=0) -> int:
    import numpy as np

    ids, X = embed.load_matrix(conn)
    if len(ids) == 0:
        log("特徴ベクトルがありません。先にメニュー8（Embed）を実行してください。")
        return 2
    pos_ix = {int(a): i for i, a in enumerate(ids)}
    zf = zeroshot.features(cfg.app_root / "models" / embed.MODEL_NAME, X)
    neg = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM photo_reviews WHERE answer='none' AND asset_id IS NOT NULL")}
    neg = {a for a in neg if a in pos_ix}
    lab = collect_labels(conn)
    # フォルダ登録（config/nopet_folders.toml）
    by_kind, missing = nopetfolders.assets_by_kind(conn, nopetfolders.load(cfg))
    user_live = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM photo_reviews WHERE answer<>'none' AND asset_id IS NOT NULL")}
    kind_of = {}
    other = set()
    for (kind, live), s in by_kind.items():
        s = {a for a in s if a in pos_ix}
        if live:
            other |= s
        else:
            s -= user_live
            neg |= s
            for a in s:
                kind_of[a] = kind
    for kind in sorted({k for k, _ in by_kind}):
        n = len({a for (k, lv), s in by_kind.items() if k == kind for a in s if a in pos_ix})
        log(f"登録フォルダ「{kind}」: {n:,} 件" + ("（生きたウサギ・「いる」側で学習）" if (kind, True) in by_kind else ""))
    for m in missing:
        log(f"  ※ DBに写真が見つからないフォルダ: {m}（パスの綴りを確認してください）")
    other -= neg
    pos = sorted(({a for a in lab if a in pos_ix} | other) - neg)
    log(f"学習に使う写真: ペットが写っていない {len(neg):,} 件 / ペットが写っている（名前の推定・確認済み）{len(pos):,} 件")
    if len(neg) < 30:
        log("「いない」と確認した写真が30枚に満たないので学習できません。review_*.csv を記入して import-photos してください。")
        return 3
    keys = sorted(neg) + pos
    y = np.array([1.0] * len(neg) + [0.0] * len(pos))
    Xt = X[[pos_ix[a] for a in keys]]
    folder = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        folder.setdefault(r["asset_id"], (r["vol_id"], r["volpath"].rsplit("/", 1)[0] if "/" in r["volpath"] else ""))
    fl = sorted({folder.get(a) for a in keys}, key=str)
    random.Random(seed).shuffle(fl)
    # 登録フォルダは、種類ごとに順に別々の分割へ入れる（偶然同じ分割に集まって、学習に1つも使えない、を防ぐ）
    reg = {}
    for a in keys:
        if a in kind_of or a in other:
            reg.setdefault(folder.get(a), kind_of.get(a, "他のウサギ"))
    fl = sorted(reg, key=lambda f: (reg[f], str(f))) + [f for f in fl if f not in reg]
    fmap = {f: i % 5 for i, f in enumerate(fl)}
    fi = np.array([fmap[folder.get(a)] for a in keys])
    wneg = min(20.0, len(pos) / len(neg))
    wt = np.where(y == 1, wneg, 1.0)
    ix_all = [pos_ix[a] for a in keys]
    cands = {"CLIPの特徴のみ": X}
    if zf is not None:
        cands["ゼロショットの点数のみ"] = zf
        cands["両方"] = np.hstack([X, zf])

    def cv_of(F):
        Ft = F[ix_all]
        c = np.zeros(len(keys))
        for k in range(5):
            tr, te = fi != k, fi == k
            if te.any() and tr.any() and len(set(y[tr])) == 2:
                c[te] = score(fit_logistic(Ft[tr], y[tr], weights=wt[tr]), Ft[te])
        return c

    live_ok = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM photo_reviews WHERE answer<>'none' AND asset_id IS NOT NULL")}
    m_other = np.array([a in other for a in keys])
    m_live = np.array([a in live_ok for a in keys])
    m_pos = y == 0
    cvs, best, best_v = {}, None, -9
    if len(cands) > 1:
        log("\n特徴の比較（しきい値0.5。検出率は高いほど、誤検出は低いほどよい）:")
    for name, F in cands.items():
        c = cv_of(F)
        cvs[name] = c
        det = float(((c >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1))
        fo = float((c[m_other] >= 0.5).mean()) if m_other.any() else 0.0
        fl = float((c[m_live] >= 0.5).mean()) if m_live.any() else 0.0
        fp = float((c[m_pos] >= 0.5).mean())
        if len(cands) > 1:
            log(f"  {name:<14} 検出率 {det * 100:5.1f}% / 他のウサギの誤判定 {fo * 100:5.1f}% / 確認済みペットの誤判定 {fl * 100:5.1f}% / 名前の推定の写真 {fp * 100:5.2f}%")
        v = det - 2 * max(fo, fl)
        if v > best_v:
            best, best_v = name, v
    if len(cands) > 1:
        log(f"→ 「{best}」を使います。")
    X = cands[best]
    Xt = X[ix_all]
    cv = cvs[best]
    log("\n学習に使っていないフォルダの写真での当たり方:")
    for t in (0.5, 0.7, 0.9):
        det = float(((cv >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))
        fp = float(((cv >= t) & (y == 0)).sum() / max((y == 0).sum(), 1))
        log(f"  しきい値 {t:.1f}: 「いない」写真の検出率 {det * 100:5.1f}% / ペットが写っている写真の誤検出率 {fp * 100:5.2f}%")
    if kind_of:
        log("\n登録フォルダの種類ごとの検出率（そのフォルダを学習に使わない場合）:")
        kk = np.array([kind_of.get(a, "") for a in keys])
        for kind in sorted(set(kind_of.values())):
            m = kk == kind
            log(f"  {kind}: " + " / ".join(f"{t:.1f}以上 {float((cv[m] >= t).mean()) * 100:5.1f}%" for t in (0.5, 0.7, 0.9)) + f"  （{int(m.sum()):,} 件）")
    model = fit_logistic(Xt, y, weights=wt)
    if other:
        ko = np.array([a in other for a in keys])
        log("\n他のウサギ（そのフォルダを学習に使わない場合）が誤って「いない」とされる率: " + " / ".join(f"{t:.1f}以上 {float((cv[ko] >= t).mean()) * 100:5.1f}%" for t in (0.5, 0.7, 0.9)) + f"  （{int(ko.sum()):,} 件）")
    real = {r["id"] for r in conn.execute("SELECT id FROM assets WHERE zone='real' AND kind='image'")}
    qa = [int(a) for a in ids if int(a) in real]
    sc = score(model, X[[pos_ix[a] for a in qa]])
    stamp = time.strftime("%Y-%m-%d")
    conn.execute("DELETE FROM asset_nopet")
    conn.executemany("INSERT INTO asset_nopet(asset_id, score, created) VALUES(?,?,?)", [(a, float(s), stamp) for a, s in zip(qa, sc)])
    conn.commit()
    from . import nonlive
    nonlive.run(conn, log=log, cfg=cfg)
    s_of = dict(zip(qa, map(float, sc)))
    reviewed = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM photo_reviews WHERE asset_id IS NOT NULL")}
    cand = [a for a in qa if a in lab and a not in reviewed]
    n9, n7, n5 = (sum(1 for a in cand if lo <= s_of[a] < hi) for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7)))
    log(f"\n名前の推定が付いている未確認の写真のうち、「いない」らしさが 0.9以上 {n9:,} 件 / 0.7〜0.9 {n7:,} 件 / 0.5〜0.7 {n5:,} 件")
    mounts = _mounts(conn)
    paths = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        paths.setdefault(r["asset_id"], (r["vol_id"], r["volpath"]))
    rng = random.Random(seed + 1)
    rows = []
    for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7)):
        pool = [a for a in cand if lo <= s_of[a] < hi]
        for a in rng.sample(pool, min(per_bin, len(pool))):
            rows.append((round(s_of[a], 2), a, abs_path(mounts, *paths[a]), ""))
    rep = cfg.db_path.parent / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    nrow = _write_csv(rep / "review_nopet.csv", ["いないらしさ", "asset_id", "パス", "正しいペット（ペット名 / どちらもいない）"], rows)
    log(f"確認用の写真 {nrow} 件 → {rep / 'review_nopet.csv'}")
    return 0
