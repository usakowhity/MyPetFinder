"""CLIP の特徴ベクトルだけで、ペットをどこまで当てられるかを検証する。

正解の代わりに「フォルダ名・ファイル名から付いた強いペット名（strong）」を使う。ただしこれは推定であって正解ではない。
そこで、各画像を「同じフォルダ以外」の画像だけを手がかりに kNN で判定し、名前と食い違う画像を一覧にする。
  ・食い違いが少ない → CLIP で個体を見分けられている（名前の付け方も正しい）
  ・食い違いが多い組（Mimi と Hana など） → 見た目で区別できない。撮影年など別の手がかりが必要
元の写真は一切変更しない。DB への書き込みもしない（CSVとレポートの出力のみ）。
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

from . import embed
from .report import _mounts, abs_path, _write_csv


def collect_labels(conn, composite=False):
    """{asset_id: (pet_id, folder)}。strong なペットがちょうど1つのアセットだけ。"""
    pets = defaultdict(set)
    for r in conn.execute(
        """SELECT f.asset_id, fp.pet_id FROM file_pets fp JOIN files f ON f.id = fp.file_id
           WHERE fp.origin='name_hint' AND fp.strength='strong' AND fp.status='hint' AND f.status='present' AND f.asset_id IS NOT NULL"""
    ):
        pets[r["asset_id"]].add(r["pet_id"])
    folder = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        folder.setdefault(r["asset_id"], (r["vol_id"], r["volpath"].rsplit("/", 1)[0] if "/" in r["volpath"] else ""))
    # 複数のペットが付いた画像は「Hana+Tama」のような組み合わせの1クラスとして扱う（一緒に写る写真は、見た目が単独とは別物のため）
    # composite=True なら「Hana+Tama」のような組み合わせを1クラスにする。既定は False（単独のペットだけ）。
    # 試したところ、「はな＆たま」フォルダの写真は Tama だけが写ったものも多く、見た目では分けられなかったため。
    out = {a: ("+".join(sorted(s)), folder.get(a)) for a, s in pets.items() if s and (composite or len(s) == 1)}
    # 人が確認した結果を優先する: 確認済みのペットがちょうど1匹ならそれを正解に、複数なら除く。「-」のフォルダも除く
    user = defaultdict(set)
    for r in conn.execute(
        """SELECT f.asset_id, fp.pet_id FROM file_pets fp JOIN files f ON f.id = fp.file_id
           WHERE fp.origin='user' AND fp.status='identified' AND f.status='present' AND f.asset_id IS NOT NULL"""
    ):
        user[r["asset_id"]].add(r["pet_id"])
    for a, s in user.items():
        if composite or len(s) == 1:
            out[a] = ("+".join(sorted(s)), folder.get(a))
        else:
            out.pop(a, None)
    # 「-」（ペットを特定しない）と答えたフォルダは正解データから外す
    from .importreview import NONE_MARKS, dir_key
    skip = {r["folder_key"] for r in conn.execute("SELECT folder_key, answer FROM folder_reviews") if r["answer"] in NONE_MARKS}
    if skip:
        mounts = _mounts(conn)
        for a in [a for a, (_, fol) in out.items() if fol and dir_key(abs_path(mounts, fol[0], fol[1])) in skip]:
            del out[a]
    return out


def knn_predict(Xr, labels, rfolders, Xq, qfolders, k=10, chunk=1500, allowed=None):
    """参照データ(Xr, labels)を使って、問い合わせ(Xq)をペットごとの得票で判定する。同じフォルダの参照は除く（自分自身も）。
    allowed: 各問い合わせについて、使ってよいペットの集合（撮影日と生存期間が合うもの）。None なら絞らない。
    絞る場合は、近傍を選ぶ前に、使えないペットの参照を外す。(予測ラベル, 得票率, 判定できたか) を返す。"""
    import numpy as np

    classes = sorted(set(labels))
    cid = np.array([classes.index(l) for l in labels])
    fmap = {f: i for i, f in enumerate(sorted(set(rfolders) | set(qfolders), key=str))}
    rf = np.array([fmap[f] for f in rfolders])
    qf = np.array([fmap[f] for f in qfolders])
    n = len(Xq)
    pred = np.full(n, -1)
    share = np.zeros(n, dtype=np.float32)
    k = min(k, max(1, len(Xr) - 1))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        S = Xq[s:e] @ Xr.T
        S[qf[s:e][:, None] == rf[None, :]] = -np.inf
        if allowed is not None:
            cls_ok = np.ones((e - s, len(classes)), dtype=bool)
            for r in range(e - s):
                al = allowed[s + r]
                if al is not None:
                    cls_ok[r] = [c in al for c in classes]
            S[~cls_ok[:, cid]] = -np.inf
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, idx, axis=1)
        for r in range(e - s):
            ok = np.isfinite(sims[r])
            if not ok.any():
                continue
            w = np.zeros(len(classes))
            for j, sv in zip(idx[r][ok], sims[r][ok]):
                w[cid[j]] += max(sv, 0) ** 3        # 似ているものほど強く効かせる
            if w.sum() <= 0:
                continue
            pred[s + r] = int(w.argmax())
            share[s + r] = w.max() / w.sum()
    return [classes[p] if p >= 0 else None for p in pred], share, pred >= 0


def knn_loo(X, labels, folders, k=10, min_class=30, chunk=1500, allowed=None):
    """参照データ自身を、同じフォルダ以外だけで判定する（検証用）。"""
    return knn_predict(X, labels, folders, X, folders, k=k, chunk=chunk, allowed=allowed)


def knn_species(Xr, labels, rfolders, Xq, qfolders, species_of, k=10, allowed=None):
    """二段階の判定。①ウサギかネズミか（見た目で決まる）②その種類の中でどの個体か（撮影日の生存期間で決まる。
    候補が複数残るときだけ、その種類の参照の中で個体ごとに投票）。Mimi と Hana のように見た目が同じ個体は、
    参照が少なくても、撮影年で決まる。(予測ラベル, 得票率, 判定できたか) を返す。"""
    import numpy as np

    sp_labels = [species_of.get(l, l) for l in labels]
    sp_allowed = None
    if allowed is not None:
        sp_allowed = [None if a is None else {species_of.get(c, c) for c in a} for a in allowed]
    sp_pred, sp_share, ok = knn_predict(Xr, sp_labels, rfolders, Xq, qfolders, k=k, allowed=sp_allowed)
    n = len(Xq)
    pred = [None] * n
    share = np.zeros(n, dtype=np.float32)
    pets_of: dict = {}
    for l in set(labels):
        pets_of.setdefault(species_of.get(l, l), set()).add(l)
    need: dict = {}
    for i in range(n):
        if not ok[i]:
            continue
        sp = sp_pred[i]
        cand = pets_of[sp] if allowed is None or allowed[i] is None else (pets_of[sp] & allowed[i])
        if not cand:
            cand = pets_of[sp]
        if len(cand) == 1:
            pred[i] = next(iter(cand))
            share[i] = sp_share[i]
        else:
            need.setdefault(sp, []).append((i, cand))
    for sp, items in need.items():
        ridx = [j for j, l in enumerate(sp_labels) if l == sp]
        qi = [i for i, _ in items]
        p2, s2, ok2 = knn_predict(Xr[ridx], [labels[j] for j in ridx], [rfolders[j] for j in ridx],
                                  Xq[qi], [qfolders[i] for i in qi], k=k, allowed=[c for _, c in items])
        for (i, cand), pp, ss, oo in zip(items, p2, s2, ok2):
            if oo:
                pred[i] = pp
                share[i] = sp_share[i] * ss
            else:                                   # 同じ種類の参照が他フォルダにない。候補のうち参照の多いほうを弱い判定として使う
                pred[i] = sorted(cand)[0]
                share[i] = 0.0
    okf = np.array([p is not None for p in pred])
    return pred, share, okf


def species_map(conn, labels) -> dict:
    """ペットID → 種類。「Hana+Tama」のような組み合わせは「rabbit+rat」のように、含む種類をまとめたものにする。"""
    base = {r["id"]: (r["species"] or r["id"]) for r in conn.execute("SELECT id, species FROM pets")}
    for l in set(labels):
        if "+" in l:
            base[l] = "+".join(sorted({base.get(p, p) for p in l.split("+")}))
    return base


def allowed_sets(conn, asset_ids, classes):
    """各アセットについて、撮影日と生存期間が合うペットの集合（撮影日が不明・低確度なら None＝絞らない）。"""
    from .pethints import PetIndex

    pi = PetIndex(conn)
    dinfo = {r["id"]: r for r in conn.execute("SELECT id, date_taken, date_confidence, date_source, date_conflict FROM assets")}
    out = []
    for a in asset_ids:
        d = dinfo.get(int(a))
        if d is None or not d["date_taken"]:
            out.append(None)
            continue
        ok = {c for c in classes
              if all(pi.strength(p_, d["date_taken"], d["date_confidence"], d["date_source"], bool(d["date_conflict"]))[0] != "weak"
                     for p_ in c.split("+"))}
        out.append(ok if ok else None)
    return out


def compare(cfg, conn, k=10, min_class=30, log=print) -> int:
    """CLIP と DINOv2 を、同じ画像・同じ正解で比べる（見た目だけ／撮影日・生存期間で絞った場合）。両方の特徴が付いた画像だけを使う。"""
    import numpy as np

    mats = {}
    for name in (embed.MODEL_NAME, embed.DINO_MODEL):
        ids, X = embed.load_matrix(conn, model=name)
        if len(ids):
            mats[name] = (ids, X)
    if len(mats) < 2:
        log("比べるには、CLIP（メニュー8）と DINOv2（メニュー19）の両方の特徴が必要です。")
        return 2
    common = set(int(a) for a in mats[embed.MODEL_NAME][0]) & set(int(a) for a in mats[embed.DINO_MODEL][0])
    lab = collect_labels(conn)
    cnt = Counter(lab[a][0] for a in common if a in lab)
    keep = {p for p, c in cnt.items() if c >= min_class}
    if len(keep) < 2:
        log(f"正解として使えるペットが2種類に満たない（{dict(cnt)}）ので比べられません。")
        return 3
    aids = sorted(a for a in common if a in lab and lab[a][0] in keep)
    labels = [lab[a][0] for a in aids]
    folders = [lab[a][1] for a in aids]
    classes = sorted(keep)
    species = species_map(conn, labels)
    allowed = allowed_sets(conn, aids, classes)
    n_restr = sum(1 for x in allowed if x is not None and len(x) < len(classes))
    log(f"\n=== ペット個体判別の比較: CLIP と DINOv2 ({len(aids):,} 件 / {len(set(folders)):,} フォルダ) ===")
    log("正解 = 名前（フォルダ名・人の確認）。判定 = 同じフォルダ以外の画像だけを手がかりにした kNN。")
    log(f"撮影日・生存期間で絞れた画像: {n_restr:,} 件")
    res = {}
    for name, (ids, X) in mats.items():
        pos = {int(a): i for i, a in enumerate(ids)}
        Xs = X[[pos[a] for a in aids]]
        pred, _, ok = knn_loo(Xs, labels, folders, k=k)
        pred2, _, ok2 = knn_species(Xs, labels, folders, Xs, folders, species, k=k, allowed=allowed)
        res[name] = (pred, ok, pred2, ok2)
    for title, idx in (("見た目だけ（撮影日・生存期間を使わない）", (0, 1)), ("撮影日・生存期間で絞った場合（実際の判定と同じ）", (2, 3))):
        log(f"\n[{title}]")
        head = "  " + f"{'':<8}" + "".join(f"{n:>16}" for n in res)
        log(head)
        rows = [("全体", None)] + [(c, c) for c in classes]
        for label, c in rows:
            line = f"  {label:<8}"
            for name, r in res.items():
                pred, ok = r[idx[0]], r[idx[1]]
                m = [(l == p) for l, p, o in zip(labels, pred, ok) if o and (c is None or l == c)]
                line += f"{(sum(m) * 100 / max(len(m), 1)):15.1f}%"
            log(line)
    return 0


def run(cfg, conn, k=10, min_class=30, model=None, log=print) -> int:
    import numpy as np

    lines = []

    def out(s=""):
        lines.append(s)
        log(s)

    model = model or embed.pet_model(conn)
    ids, X = embed.load_matrix(conn, model=model)
    out(f"（特徴: {'DINOv2' if model == embed.DINO_MODEL else 'CLIP'}）")
    if len(ids) == 0:
        out("特徴ベクトルがありません。先にメニューの「Embed」を実行してください。")
        return 2
    lab = collect_labels(conn)
    rows = [(a, i) for i, a in enumerate(ids) if int(a) in lab]
    cnt = Counter(lab[int(a)][0] for a, _ in rows)
    keep = {p for p, c in cnt.items() if c >= min_class}
    rows = [(a, i) for a, i in rows if lab[int(a)][0] in keep]
    if len(keep) < 2:
        out(f"正解として使えるペットが2種類に満たない（{dict(cnt)}）ので検証できません。")
        return 3
    sel = np.array([i for _, i in rows])
    aids = [int(a) for a, _ in rows]
    labels = [lab[a][0] for a in aids]
    folders = [lab[a][1] for a in aids]
    out(f"=== ペット判別の検証 ({len(aids):,} 件 / {len(set(folders)):,} フォルダ) ===")
    out("正解 = フォルダ名・ファイル名から付いた強いペット名。判定 = 同じフォルダ以外の画像だけを手がかりにした kNN。")
    pred, share, ok = knn_loo(X[sel], labels, folders, k=k)

    n_ok = int(ok.sum())
    out(f"\n判定できた画像: {n_ok:,} / {len(aids):,}")
    classes = sorted(keep)
    conf = {c: Counter() for c in classes}
    for l, p, o in zip(labels, pred, ok):
        if o:
            conf[l][p] += 1
    tot_ok = sum(conf[c][c] for c in classes)
    out(f"全体の一致率: {tot_ok * 100 / max(n_ok, 1):.1f}%\n")
    out("[ペットごと]  名前の付いた画像のうち、CLIPも同じペットと判定した割合（再現率）と、混同先")
    for c in classes:
        t = sum(conf[c].values())
        if not t:
            continue
        others = ", ".join(f"{p} {v:,}" for p, v in conf[c].most_common() if p != c and v)
        out(f"  {c:<8} {conf[c][c] * 100 / t:5.1f}%  ({conf[c][c]:,}/{t:,})   食い違い: {others or 'なし'}")
    out("\n[判定の自信度ごとの一致率]  得票率が高いほど当たるか")
    for lo, hi in ((0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.0, 0.6)):
        m = [(l == p) for l, p, o, s in zip(labels, pred, ok, share) if o and lo <= s < hi]
        if m:
            out(f"  得票率 {lo:.1f}〜{min(hi, 1):.1f}: {sum(m) * 100 / len(m):5.1f}%  ({len(m):,}件)")

    species = species_map(conn, labels)
    sp_l = [species.get(l, l) for l in labels]
    # 撮影日と生存期間で絞った判定（撮影日が信頼できる画像だけ。期間外のペットは候補から外す）
    allowed = allowed_sets(conn, aids, classes)
    n_restr = sum(1 for x in allowed if x is not None and len(x) < len(classes))
    species_of = species

    pred2, share2, ok2 = knn_species(X[sel], labels, folders, X[sel], folders, species_of, k=k, allowed=allowed)
    out(f"\n[撮影日と生存期間で絞った場合]  候補を絞れた画像 {n_restr:,} 件（撮影日が信頼でき、期間外のペットがあるもの）")
    t2 = sum(1 for l, p, o in zip(labels, pred2, ok2) if o and l == p)
    out(f"  全体の一致率: {t2 * 100 / max(int(ok2.sum()), 1):.1f}%   （絞る前 {tot_ok * 100 / max(n_ok, 1):.1f}%）")
    for c in classes:
        tt = [p for l, p, o in zip(labels, pred2, ok2) if o and l == c]
        if tt:
            wrong = Counter(p for p in tt if p != c)
            out(f"  {c:<8} {sum(p == c for p in tt) * 100 / len(tt):5.1f}%  ({sum(p == c for p in tt):,}/{len(tt):,})   食い違い: "
                + (", ".join(f"{k_} {v:,}" for k_, v in wrong.most_common()) or "なし"))
    for lo, hi in ((0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.0, 0.6)):
        m = [(l == p) for l, p, o, sh in zip(labels, pred2, ok2, share2) if o and lo <= sh < hi]
        if m:
            out(f"  得票率 {lo:.1f}〜{min(hi, 1):.1f}: {sum(m) * 100 / len(m):5.1f}%  ({len(m):,}件)")
    # 取り違えの中身を見る: 名前は A なのに AI は B（撮影日で絞ったあと）。フォルダごとの件数と、全件の一覧
    (cfg.db_path.parent / "reports").mkdir(parents=True, exist_ok=True)
    mounts0 = _mounts(conn)
    paths0 = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        paths0.setdefault(r["asset_id"], (r["vol_id"], r["volpath"]))
    dinfo0 = {r["id"]: r for r in conn.execute("SELECT id, date_taken, date_confidence, date_source FROM assets")}
    errs = [(a, l, p, sh) for a, l, p, o, sh in zip(aids, labels, pred2, ok2, share2) if o and p != l]
    for (lab_a, lab_b) in [k for k, _ in Counter((e[1], e[2]) for e in errs).most_common(2)]:
        sub = [e for e in errs if e[1] == lab_a and e[2] == lab_b]
        if not sub:
            continue
        byf = Counter(os.path.dirname(abs_path(mounts0, *paths0[a])) for a, _, _, _ in sub)
        dated = sum(1 for a, _, _, _ in sub if dinfo0[a]["date_taken"] and dinfo0[a]["date_confidence"] in ("high", "medium"))
        out(f"\n[{lab_a} なのに {lab_b} と判定された {len(sub):,} 件の内訳]  撮影日が信頼できるもの {dated:,} 件")
        for f, c in byf.most_common(8):
            out(f"  {c:4,} 件  {f}")
        _write_csv(cfg.db_path.parent / "reports" / f"errors_{lab_a}_as_{lab_b}.csv",
                   ["フォルダ", "パス", "AIの自信", "撮影日", "撮影日の出どころ", "正しいペット"],
                   sorted((os.path.dirname(abs_path(mounts0, *paths0[a])), abs_path(mounts0, *paths0[a]), round(float(sh), 2),
                           dinfo0[a]["date_taken"] or "", dinfo0[a]["date_source"] or "", "") for a, _, _, sh in sub))
    sp_p2 = [species.get(p, p) for p in pred2]
    out(f"  種類（ウサギ／ネズミ）だけで見た一致率: {sum(a == b for a, b, o in zip(sp_l, sp_p2, ok2) if o) * 100 / max(int(ok2.sum()), 1):.1f}%")

    species = species_map(conn, labels)
    sp_l = [species.get(l, l) for l in labels]
    sp_p = [species.get(p, p) for p in pred]
    sp_ok = [a == b for a, b, o in zip(sp_l, sp_p, ok) if o]
    out(f"\n[種類（ウサギ／ネズミ）だけで見た一致率] {sum(sp_ok) * 100 / max(len(sp_ok), 1):.1f}%")
    for sp in sorted(set(sp_l)):
        t = [(a, b) for a, b, o in zip(sp_l, sp_p, ok) if o and a == sp]
        wrong = Counter(b for a, b in t if a != b)
        out(f"  {sp:<8} {sum(a == b for a, b in t) * 100 / max(len(t), 1):5.1f}%  ({len(t):,}件)  食い違い: "
            + (", ".join(f"{k} {v:,}" for k, v in wrong.most_common()) or "なし"))

    mounts = _mounts(conn)
    paths = {}
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        paths.setdefault(r["asset_id"], (r["vol_id"], r["volpath"]))
    bad = []
    for a, l, p, s, o in zip(aids, labels, pred, share, ok):
        if o and p != l and s >= 0.7:
            vid, vp = paths[a]
            bad.append((round(float(s), 2), a, l, p, abs_path(mounts, vid, vp)))
    bad.sort(key=lambda t: -t[0])
    rep = cfg.db_path.parent / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    nrow = _write_csv(rep / "review_ai_pets.csv", ["自信度", "asset_id", "名前から(現在)", "AIの判定", "パス", "正しいペット"],
                      [(s, a, l, p, path, "") for s, a, l, p, path in bad])
    out(f"\n名前とAIの判定が食い違い、AIの自信が高い画像（得票率0.7以上）: {nrow:,} 件")
    out(f"  → {rep / 'review_ai_pets.csv'}  （自信度の高い順。目で見て「正しいペット」欄に書けます）")
    # フォルダ単位: 名前と判定が多数食い違うフォルダ（フォルダ名の付け方の誤り・別のペットが混ざっている可能性）
    fol = defaultdict(lambda: [0, 0, Counter(), Counter(), None])
    for a, l, p, o in zip(aids, labels, pred, ok):
        if not o:
            continue
        f = lab[a][1]
        rec = fol[f]
        rec[0] += 1
        rec[1] += (p != l)
        rec[2][l] += 1
        rec[3][p] += 1
        rec[4] = rec[4] or paths[a]
    frows = []
    from .importreview import dir_key
    reviewed = {r["folder_key"] for r in conn.execute("SELECT folder_key FROM folder_reviews")}
    for f, (n, bad_n, nl, npred, ex) in fol.items():
        if n >= 5 and bad_n / n >= 0.5:
            vid, vp = ex
            if dir_key(os.path.dirname(abs_path(mounts, vid, vp))) in reviewed:
                continue          # 確認済みのフォルダは、もう一覧に出さない
            frows.append((n, bad_n, nl.most_common(1)[0][0], npred.most_common(1)[0][0],
                          f"{bad_n * 100 // n}%", os.path.dirname(abs_path(mounts, vid, vp)), "", ))
    frows.sort(key=lambda t: -t[1])
    nf = _write_csv(rep / "review_ai_folders.csv", ["枚数", "食い違い枚数", "名前から(現在)", "AIの多数判定", "食い違い率", "フォルダ", "正しいペット"],
                    frows)
    out(f"名前と判定が半分以上食い違うフォルダ: {nf:,} 件  → {rep / 'review_ai_folders.csv'}")
    (rep / "petcheck.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0
