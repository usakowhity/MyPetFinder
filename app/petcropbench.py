"""検証: ペットの個体を見分けるとき、「写真全体」より「動物の部分を切り出したもの」の方が当たるか。

一般公開用アプリの設計（ペットも人物と同じ「グループに名前を付ける」方式にする）の前提を、あなたのデータで確かめる。
  ・正解 = メニュー9と同じ（フォルダ名から付いた強いペット名と、人が確認した結果）。
  ・判定 = 同じフォルダ以外の画像だけを手がかりにした kNN（メニュー9と同じ）。撮影日・生存期間は使わない（見た目だけの力を比べるため）。
  ・比べるもの:
      CLIP 全体   : いま使っている方式（写真全体の特徴）
      CLIP 切り出し: 1枚を19通りの窓（全体＋大小の窓）に切り、「rabbit」「rat」などの文章に最も近い窓の特徴
      DINOv2 全体 / 切り出し: 個体の見分けに強いとされる別のモデル（初回に約90MBを取得。取得できなければ省く）
  ・名前付けの手間の見積り: 似たものを自動でグループにまとめたときの、グループの数・純度（同じペットだけのグループの割合）
元の写真は変更しない。DBへの書き込みもしない。結果は data/reports/petcrop_bench.txt と petcrop_samples.jpg（切り出し位置の確認用）。
"""
from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bench, embed
from .petcheck import collect_labels, knn_loo, species_map
from .report import _mounts, abs_path

MAXSIDE = 1024
DINO_URL = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def windows(w: int, h: int) -> list:
    """全体と、大小の窓（0.6倍を3×3、0.4倍を3×3）の座標 (x0,y0,x1,y1)。先頭が全体。"""
    boxes = [(0, 0, w, h)]
    for s in (0.6, 0.4):
        for iy in range(3):
            for ix in range(3):
                fx, fy = (1 - s) * ix / 2, (1 - s) * iy / 2
                boxes.append((int(fx * w), int(fy * h), max(int((fx + s) * w), int(fx * w) + 2), max(int((fy + s) * h), int(fy * h) + 2)))
    return boxes


def pick_sample(rows, per_pet, seed=0):
    """ペットごとに最大 per_pet 件。フォルダが偏らないよう、フォルダを順番に回して選ぶ。rows = [(aid, label, folder)]"""
    import random

    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    for a, l, f in rows:
        by[l][f].append(a)
    out = []
    for l, fol in by.items():
        for v in fol.values():
            rng.shuffle(v)
        order = sorted(fol, key=str)
        rng.shuffle(order)
        got = []
        while len(got) < per_pet and any(fol[f] for f in order):
            for f in order:
                if fol[f] and len(got) < per_pet:
                    got.append(fol[f].pop())
        out += got
    return sorted(out)


def _load_rgb(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.draft("RGB", (MAXSIDE * 2, MAXSIDE * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAXSIDE, MAXSIDE))
        return im.copy()


def _prep_windows(path):
    try:
        import numpy as np

        im = _load_rgb(path)
        w, h = im.size
        boxes = windows(w, h)
        arr = np.stack([bench._prep_pil(im.crop(b)) for b in boxes])
        return im, boxes, arr
    except Exception:                                                  # noqa: BLE001
        return None


def _unit(e):
    import numpy as np

    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


class Dino:
    """検証用の薄い入れ物（本体は dino.Encoder）。"""

    def __init__(self, path):
        from . import dino

        self.enc = dino.Encoder(path)

    def embed(self, ims):
        return self.enc.embed_images(ims)


def leader_clusters(X, thr, seed=0):
    """似たものを順にグループへ入れる（グループの先頭と thr 以上似ていれば同じグループ）。各画像のグループ番号を返す。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    leaders, assign = [], np.zeros(len(X), dtype=int)
    L = np.zeros((0, X.shape[1]), dtype=X.dtype)
    for i in order:
        if len(L):
            s = L @ X[i]
            j = int(s.argmax())
            if s[j] >= thr:
                assign[i] = j
                continue
        assign[i] = len(leaders)
        leaders.append(i)
        L = np.vstack([L, X[i][None, :]])
    return assign


def cluster_table(X, labels, thrs, min_size=5):
    """グループの数（min_size 以上）、その中に入った画像の割合、そのグループの純度（グループ内で最多のペットの割合）。"""
    rows = []
    for t in thrs:
        a = leader_clusters(X, t)
        groups = defaultdict(list)
        for i, g in enumerate(a):
            groups[g].append(labels[i])
        big = [v for v in groups.values() if len(v) >= min_size]
        n_in = sum(len(v) for v in big)
        pure = sum(Counter(v).most_common(1)[0][1] for v in big)
        rows.append((t, len(big), n_in / max(len(X), 1), pure / max(n_in, 1)))
    return rows


def evaluate(name, X, labels, folders, species_of, k, out):
    import numpy as np

    classes = sorted(set(labels))
    pred, share, ok = knn_loo(X, labels, folders, k=k)
    n_ok = int(ok.sum())
    acc = sum(1 for l, p, o in zip(labels, pred, ok) if o and l == p) * 100 / max(n_ok, 1)
    out(f"\n■ {name}   全体の一致率 {acc:.1f}%  （判定できた {n_ok:,} / {len(labels):,} 件）")
    conf = {c: Counter() for c in classes}
    for l, p, o in zip(labels, pred, ok):
        if o:
            conf[l][p] += 1
    for c in classes:
        t = sum(conf[c].values())
        if t:
            oth = ", ".join(f"{p} {v:,}" for p, v in conf[c].most_common() if p != c and v)
            out(f"   {c:<8} 再現率 {conf[c][c] * 100 / t:5.1f}%  ({conf[c][c]:,}/{t:,})  食い違い: {oth or 'なし'}")
    # 同じ種類の中だけで、個体を見分けられるか（Kuro と Usako、Chuchu と Momo など）
    by_sp = defaultdict(list)
    for c in classes:
        by_sp[species_of.get(c, c)].append(c)
    for sp, cs in sorted(by_sp.items()):
        if len(cs) < 2:
            continue
        idx = [i for i, l in enumerate(labels) if l in cs]
        p2, _, o2 = knn_loo(X[idx], [labels[i] for i in idx], [folders[i] for i in idx], k=k)
        n2 = int(o2.sum())
        a2 = sum(1 for i, p, o in zip(idx, p2, o2) if o and labels[i] == p) * 100 / max(n2, 1)
        out(f"   [{sp} の中で個体を当てる: {' / '.join(cs)}]  一致率 {a2:.1f}%  （{n2:,} 件）")
    return acc


def run(cfg, conn, per_pet=250, k=10, min_class=30, dino=True, log=print) -> int:
    import numpy as np
    import onnxruntime as ort

    lines = []

    def out(s=""):
        lines.append(s)
        log(s)

    ids, XW = embed.load_matrix(conn)
    if len(ids) == 0:
        out("特徴ベクトルがありません。先にメニュー8（Embed）を実行してください。")
        return 2
    lab = collect_labels(conn)
    cnt = Counter(v[0] for a, v in lab.items())
    keep = {p for p, c in cnt.items() if c >= min_class}
    if len(keep) < 2:
        out(f"正解として使えるペットが2種類に満たない（{dict(cnt)}）ので検証できません。")
        return 3
    have = {int(a) for a in ids}
    rows = [(a, v[0], v[1]) for a, v in lab.items() if v[0] in keep and a in have]
    sel = pick_sample(rows, per_pet)
    lab_of = {a: (l, f) for a, l, f in rows}
    mounts = _mounts(conn)
    paths = {}
    sset = set(sel)
    for r in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL ORDER BY id"):
        if r["asset_id"] in sset:
            p = abs_path(mounts, r["vol_id"], r["volpath"])
            if os.path.exists(p):
                paths[r["asset_id"]] = p
    sel = [a for a in sel if a in paths]
    species = species_map(conn, [lab_of[a][0] for a in sel])
    out(f"=== ペット個体の判別: 写真全体 と 切り出し の比較 ({len(sel):,} 件 / {len(set(lab_of[a][1] for a in sel)):,} フォルダ) ===")
    out("ペットごとの件数: " + ", ".join(f"{c} {n}" for c, n in sorted(Counter(lab_of[a][0] for a in sel).items())))
    out("判定 = 同じフォルダ以外の画像だけを手がかりにした kNN（撮影日・生存期間は使わない）")

    # 文章（動物の種類）
    from . import zeroshot

    model_dir = cfg.app_root / "models" / embed.MODEL_NAME
    zeroshot.ensure_files(model_dir, log)
    sp_names = sorted({species.get(lab_of[a][0], "animal") for a in sel})
    prompts = [f"a photo of a {s}" for s in sp_names]
    T = zeroshot.text_embeddings(model_dir, prompts, log)

    mp = embed.model_path(cfg, "fp32", log)
    sess = ort.InferenceSession(str(mp), ort.SessionOptions(), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    dn = None
    if dino:
        try:
            dp = cfg.app_root / "models" / "dinov2-small" / "model.onnx"
            if not (dp.exists() and dp.stat().st_size > 1_000_000):
                out("  DINOv2 のモデルを取得します（約90MB）…")
                bench.download(DINO_URL, dp, log)
            dn = Dino(dp)
        except Exception as ex:                                        # noqa: BLE001
            out(f"  （DINOv2 を使えないため、CLIP だけで比べます: {type(ex).__name__}: {str(ex)[:120]}）")
            dn = None

    feats = defaultdict(list)
    chosen = []
    n_full = 0
    workers = max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        step = max(workers, 6)
        for s in range(0, len(sel), step):
            part = sel[s: s + step]
            for a, got in zip(part, ex.map(lambda x: _prep_windows(paths[x]), part)):
                if got is None:
                    feats["_bad"].append(a)
                    continue
                im, boxes, arr = got
                E = _unit(bench._pick_embedding(sess.run(outs, {inp: arr})).astype(np.float32))
                score = (E @ T.T).max(axis=1)                             # 各窓の「動物らしさ」（種類の文章との近さの最大）
                b = int(score.argmax())
                n_full += int(b == 0)
                chosen.append((a, boxes[b], im.size))
                feats["clip_whole"].append(E[0])
                feats["clip_crop"].append(E[b])
                feats["ids"].append(a)
                if dn is not None:
                    try:
                        D = dn.embed([im, im.crop(boxes[b])])
                        feats["dino_whole"].append(D[0])
                        feats["dino_crop"].append(D[1])
                    except Exception as e:                              # noqa: BLE001
                        out(f"  （DINOv2 の実行に失敗したため省きます: {type(e).__name__}: {str(e)[:120]}）")
                        dn = None
                        for key in ("dino_whole", "dino_crop"):
                            feats.pop(key, None)
                done += 1
            el = time.time() - t0
            log(f"  特徴を計算中 {done:,}/{len(sel):,}  {done / max(el, 1e-9):.1f}枚/秒  残り約{bench.fmt_time((len(sel) - done) / max(done / max(el, 1e-9), 1e-9))}")
    ok_ids = feats["ids"]
    if len(ok_ids) < 2 * min_class:
        out("画像を読めたものが少なすぎます。")
        return 4
    labels = [lab_of[a][0] for a in ok_ids]
    folders = [lab_of[a][1] for a in ok_ids]
    out(f"\n切り出しで「全体」が選ばれた割合: {n_full * 100 / len(ok_ids):.1f}%（残りは、窓を切り出した方が動物らしいと判定）")
    names = {"clip_whole": "CLIP 全体（いまの方式）", "clip_crop": "CLIP 切り出し", "dino_whole": "DINOv2 全体", "dino_crop": "DINOv2 切り出し"}
    mats = {}
    summary = []
    for key in ("clip_whole", "clip_crop", "dino_whole", "dino_crop"):
        if len(feats.get(key, [])) == len(ok_ids):
            mats[key] = np.stack(feats[key]).astype(np.float32)
            summary.append((names[key], evaluate(names[key], mats[key], labels, folders, species, k, out)))
    out("\n=== 全体の一致率のまとめ（高いほど良い）===")
    for n, a in summary:
        out(f"  {n:<24} {a:5.1f}%")

    out("\n=== 名前付けの手間の見積り ===")
    out("似たものを自動でグループにまとめ、5件以上のグループだけを数える。「純度」= グループの中で最も多いペットの割合（高いほど、グループ全体に1つの名前を付けて済む）")
    for key, m in mats.items():
        sims = (m[:400] @ m[:400].T)[np.triu_indices(min(400, len(m)), 1)]
        lo, hi = float(np.quantile(sims, 0.5)), float(np.quantile(sims, 0.98))
        thrs = [round(lo + (hi - lo) * f, 3) for f in (0.3, 0.5, 0.7, 0.85)]
        out(f"\n■ {names[key]}")
        out("   しきい値  グループ数  グループに入った割合  純度")
        for t, n, cov, pur in cluster_table(m, labels, thrs):
            out(f"   {t:6.3f}   {n:8,}   {cov * 100:12.1f}%   {pur * 100:6.1f}%")
    rep = Path(cfg.db_path).parent / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "petcrop_bench.txt").write_text("\n".join(lines), encoding="utf-8")
    try:
        _sheet(paths, chosen, rep / "petcrop_samples.jpg")
        out(f"\n切り出し位置の確認用の画像: {rep / 'petcrop_samples.jpg'}  （赤い枠 = 選ばれた窓）")
    except Exception as ex:                                              # noqa: BLE001
        out(f"（確認用の画像は作れませんでした: {ex}）")
    out(f"結果を保存しました: {rep / 'petcrop_bench.txt'}")
    return 0


def _sheet(paths, chosen, dest, n=24, cols=6, cell=220):
    """切り出し位置の確認用に、ランダムな n 枚へ赤い枠を描いた一覧を作る。"""
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(1)
    pick = rng.sample(chosen, min(n, len(chosen)))
    rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    for i, (a, box, size) in enumerate(pick):
        im = _load_rgb(paths[a])
        w, h = im.size
        sc = (cell - 4) / max(w, h)
        th = im.resize((max(1, round(w * sc)), max(1, round(h * sc))))
        d = ImageDraw.Draw(th)
        d.rectangle([box[0] * sc, box[1] * sc, box[2] * sc - 1, box[3] * sc - 1], outline=(255, 0, 0), width=3)
        sheet.paste(th, ((i % cols) * cell + 2, (i // cols) * cell + 2))
    sheet.save(dest, quality=85)
