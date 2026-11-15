"""DINOv2（small）の画像特徴。ペットの「個体」を見分けるために使う（CLIP は文章検索・場所・場面に使い続ける）。

検証（メニュー18）では、同じペットの別の写真を探す力が、CLIP より高かった（全体の一致率 +5pt、グループの純度 +12〜20pt）。
 ・入力: 短い辺を256に縮小 → 中央224角 → ImageNet の平均・標準偏差（DINOv2 の標準の前処理）
 ・出力: 384個の数値（まとめの特徴）。L2正規化して float16 で保存する。embeddings テーブルに、CLIP とは別のキー（model='dinov2-small'）で入る。
 ・文章には対応しない。モデルは onnx-community/dinov2-small（元は Meta の DINOv2）。公開時はライセンスの確認が必要。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bench, embed

MODEL_NAME = "dinov2-small"
URL = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def model_path(cfg, log=print) -> Path:
    p = cfg.app_root / "models" / MODEL_NAME / "model.onnx"
    if not (p.exists() and p.stat().st_size > 1_000_000):
        log("  DINOv2 のモデルを取得します（約90MB）…")
        bench.download(URL, p, log)
    return p


def prep_pil(im):
    return bench._prep_pil(im, MEAN, STD, resize=256, crop=224)


def preprocess(path: str):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.draft("RGB", (448, 448))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return prep_pil(im)


class Encoder:
    def __init__(self, path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), ort.SessionOptions(), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def encode(self, arrays):
        """前処理済みの配列（N,3,224,224）→ L2正規化した (N, 384)。"""
        import numpy as np

        outs = self.sess.run(None, {self.inp: np.asarray(arrays, dtype=np.float32)})
        two = next((o for o in outs if getattr(o, "ndim", 0) == 2), None)
        e = two if two is not None else outs[0][:, 0]                   # 2次元の出力（まとめの特徴）か、先頭トークン
        e = e.astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

    def embed_images(self, ims):
        import numpy as np

        return self.encode(np.stack([prep_pil(im) for im in ims]))


def run(cfg, conn, limit=None, retry=False, model=None, batch=8, chunk=64, log=print) -> int:
    """実写の全画像に DINOv2 の特徴を付ける（途中で止めても続きから再開できる）。"""
    import numpy as np

    from .report import _mounts

    try:
        mp = Path(model) if model else model_path(cfg, log)
    except Exception as ex:
        log(f"モデルを取得できませんでした: {ex}")
        return 3
    todo = embed.pending_assets(conn, _mounts(conn), limit, retry, model=MODEL_NAME)
    done_before = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model=? AND vec IS NOT NULL", (MODEL_NAME,)).fetchone()[0]
    log(f"DINOv2 の特徴を付ける画像: {len(todo):,} 件（済み {done_before:,} 件）")
    if not todo:
        return 0
    enc = Encoder(mp)

    def prep(p):
        try:
            return preprocess(p)
        except Exception as ex:                                           # noqa: BLE001
            return ex

    t0, n_ok, n_bad = time.time(), 0, 0
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
            for i in range(0, len(todo), chunk):
                part = todo[i:i + chunk]
                res = list(ex.map(prep, [p for _, p in part]))
                ok = [(aid, a) for (aid, _), a in zip(part, res) if not isinstance(a, Exception)]
                rows = [(aid, MODEL_NAME, 0, None, f"{type(a).__name__}: {a}"[:200], stamp)
                        for (aid, _), a in zip(part, res) if isinstance(a, Exception)]
                n_bad += len(rows)
                embs = [enc.encode(np.stack([a for _, a in ok[j:j + batch]])) for j in range(0, len(ok), batch)]
                if embs:
                    E = np.concatenate(embs).astype(np.float16)
                    rows += [(aid, MODEL_NAME, int(v.shape[0]), v.tobytes(), None, stamp) for (aid, _), v in zip(ok, E)]
                    n_ok += len(ok)
                conn.executemany("INSERT OR REPLACE INTO embeddings(asset_id,model,dim,vec,err,created) VALUES(?,?,?,?,?,?)", rows)
                conn.commit()
                el = time.time() - t0
                k = i + len(part)
                log(f"  {k:,}/{len(todo):,}  {k / el:5.1f}枚/秒  残り約{bench.fmt_time(el / k * (len(todo) - k))}  (失敗 {n_bad})")
    except KeyboardInterrupt:
        conn.commit()
        log("中断しました。もう一度実行すると続きから再開します。")
        return 1
    log(f"完了: 成功 {n_ok:,} 件 / 失敗 {n_bad:,} 件 / {bench.fmt_time(time.time() - t0)}")
    if n_ok:
        try:
            from . import petcheck

            petcheck.compare(cfg, conn, log=log)
        except Exception as ex:                                           # noqa: BLE001
            log(f"（精度の比較は省きました: {type(ex).__name__}: {ex}）")
    return 0
