"""ネズミらしさ（rat_score）: 写真が「ネズミっぽい」か「ウサギっぽい」かを 0〜1 で出す。

ネズミだけ／ウサギだけの時代（他の種類が生きていない年）の写真だけで学習した線形モデルなので、
Momo と Usako が一緒にいた時代の写真でも、フォルダ名の誤りに引きずられない。
  ・ネズミだけの時代（例: Chuchu 2004〜2005年）→ 1、ウサギだけの時代（例: Usako 2007年、2011年以降）→ 0
  ・両方が写る写真は中間の値になりやすい（学習には両方写った写真が無いので、あくまで目安）
Momo の時代の写真から、層ごとに無作為抽出して review_momo_era.csv に出す。「正しい」欄に書いてもらい、しきい値の調整に使う。
"""
from __future__ import annotations

import random
import time
from collections import defaultdict

from . import embed
from .pethints import PetIndex
from .petcheck import collect_labels, species_map
from .report import _mounts, abs_path, _write_csv

_SHOOT = ("exif", "exif_fixed", "filename")


def fit_logistic(X, y, lam=3.0, iters=15, weights=None):
    """L2つきロジスティック回帰（ニュートン法）。(重み, 平均, 標準偏差) を返す。weights は標本ごとの重み。"""
    import numpy as np

    wt = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=float)

    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    R = np.eye(Z.shape[1]) * lam
    R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (wt * (p - y)) + R @ w
        H = (Z * (wt * p * (1 - p))[:, None]).T @ Z + R + np.eye(Z.shape[1]) * 1e-6
        w -= np.linalg.solve(H, g)
    return w, mu, sd


def score(model, X):
    import numpy as np

    w, mu, sd = model
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))


def alive_species(pi, sp, date_taken) -> set:
    """その日（日が分からなければ年）に生きていたペットの種類の集合。日単位の生存期間（pets.toml の start_date / end_date）があれば、日で判断する。"""
    year = int(date_taken[:4])
    day = date_taken[:10] if len(date_taken) >= 10 else None
    out = set()
    for p, (s0, e0) in pi.periods.items():
        sd, ed = pi.exact.get(p, (None, None))
        if day and (sd or ed):
            ok = (sd is None or sd <= day) and (ed is None or day <= ed)
        else:
            ok = (s0 is None or s0 <= year) and (e0 is None or year <= e0)
        if ok:
            out.add(sp.get(p, p))
    return out


def pure_era_samples(conn, ids):
    """{asset_id: (y, folder)}: ネズミだけ(1)・ウサギだけ(0)の時代の写真。日付が信頼できるものだけ。"""
    lab = collect_labels(conn)
    pi = PetIndex(conn)
    sp = species_map(conn, list({v[0] for v in lab.values()}))
    dinfo = {r["id"]: r for r in conn.execute("SELECT id, date_taken, date_confidence, date_source, date_conflict FROM assets")}
    out = {}
    for a in ids:
        a = int(a)
        if a not in lab or "+" in lab[a][0]:
            continue
        d = dinfo.get(a)
        if not d or not d["date_taken"] or d["date_confidence"] not in ("high", "medium") or d["date_conflict"] \
                or d["date_source"] not in _SHOOT:
            continue
        alive = alive_species(pi, sp, d["date_taken"])
        mine = sp.get(lab[a][0], lab[a][0])
        if alive == {mine} and mine in ("rat", "rabbit"):
            out[a] = (1.0 if mine == "rat" else 0.0, lab[a][1])
    return out


def run(cfg, conn, log=print, per_bin=8, seed=0) -> int:
    import numpy as np

    ids, X = embed.load_matrix(conn)
    if len(ids) == 0:
        log("特徴ベクトルがありません。先にメニュー8（Embed）を実行してください。")
        return 2
    pos = {int(a): i for i, a in enumerate(ids)}
    pure = pure_era_samples(conn, ids)
    n_rat = sum(1 for y, _ in pure.values() if y == 1)
    n_rab = len(pure) - n_rat
    log(f"学習に使う写真: ネズミだけの時代 {n_rat:,} 件 / ウサギだけの時代 {n_rab:,} 件")
    if n_rat < 30 or n_rab < 30:
        log("学習できる写真が少なすぎます（ペットの生存期間・撮影日・名前の推定を確認してください）。")
        return 3
    keys = list(pure)
    y = np.array([pure[a][0] for a in keys])
    Xt = X[[pos[a] for a in keys]]
    # フォルダ単位の5分割で、学習に使っていない写真への当たり方を見る
    folders = sorted({pure[a][1] for a in keys}, key=str)
    rng = random.Random(seed)
    rng.shuffle(folders)
    fold = {f: i % 5 for i, f in enumerate(folders)}
    fi = np.array([fold[pure[a][1]] for a in keys])
    cv = np.zeros(len(keys))
    for k in range(5):
        tr, te = fi != k, fi == k
        if te.any() and tr.any() and len(set(y[tr])) == 2:
            cv[te] = score(fit_logistic(Xt[tr], y[tr]), Xt[te])
    acc_rat = float(((cv >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1))
    acc_rab = float(((cv < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1))
    log(f"学習に使っていないフォルダの写真での一致率: ネズミ {acc_rat * 100:.1f}% / ウサギ {acc_rab * 100:.1f}%")
    model = fit_logistic(Xt, y)
    real = {r["id"] for r in conn.execute("SELECT id FROM assets WHERE zone='real' AND kind='image'")}
    qa = [int(a) for a in ids if int(a) in real]
    sc = score(model, X[[pos[a] for a in qa]])
    stamp = time.strftime("%Y-%m-%d")
    conn.execute("DELETE FROM asset_presence")
    conn.executemany("INSERT INTO asset_presence(asset_id, rat_score, created) VALUES(?,?,?)",
                     [(a, float(s), stamp) for a, s in zip(qa, sc)])
    conn.commit()
    score_of = dict(zip(qa, map(float, sc)))

    # Momo の時代（ネズミとウサギが両方いた時代）の写真の分布
    pi = PetIndex(conn)
    spm = species_map(conn, [])
    base = {r["id"]: (r["species"] or r["id"]) for r in conn.execute("SELECT id, species FROM pets")}
    both = []
    for a in qa:
        d = conn.execute("SELECT date_taken, date_confidence, date_source, date_conflict FROM assets WHERE id=?", (a,)).fetchone()
        if not d or not d["date_taken"] or d["date_confidence"] not in ("high", "medium") or d["date_conflict"] \
                or d["date_source"] not in _SHOOT:
            continue
        alive = alive_species(pi, base, d["date_taken"])
        if {"rat", "rabbit"} <= alive:
            both.append(a)
    log(f"\nネズミとウサギが両方いた時代の写真: {len(both):,} 件。ネズミらしさの分布:")
    bins = defaultdict(list)
    for a in both:
        bins[min(int(score_of[a] * 10), 9)].append(a)
    for b in range(10):
        log(f"  {b / 10:.1f}〜{(b + 1) / 10:.1f}: {len(bins[b]):6,} 件  " + "#" * (len(bins[b]) * 50 // max(len(both), 1)))
    # 確認用: 層ごとに無作為抽出
    mounts = _mounts(conn)
    paths = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        paths.setdefault(r["asset_id"], (r["vol_id"], r["volpath"]))
    rows = []
    r2 = random.Random(seed + 1)
    for b in range(10):
        pick = r2.sample(bins[b], min(per_bin, len(bins[b])))
        for a in pick:
            vid, vp = paths[a]
            rows.append((round(score_of[a], 2), a, abs_path(mounts, vid, vp), ""))
    rep = cfg.db_path.parent / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    nrow = _write_csv(rep / "review_momo_era.csv", ["ネズミらしさ", "asset_id", "パス", "写っているのは（Momo / Usako / 両方 / どちらもいない）"], rows)
    log(f"\n確認用の写真 {nrow} 件 → {rep / 'review_momo_era.csv'}")
    log("  ネズミらしさの層ごとに無作為に選んであります。実際に何が写っているかを右端の欄に書いてください。")
    return 0
