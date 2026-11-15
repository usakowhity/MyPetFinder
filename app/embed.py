"""Step 2: 実写の画像すべてに CLIP の特徴ベクトルを付けて DB に保存する（途中で止めても続きから再開できる）。

元の写真は読むだけ。結果は embeddings テーブル（アセット単位・float16）に入る。
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bench

MODEL_NAME = "clip-vit-b32"
VIDEO_MODEL = MODEL_NAME + "-video"      # 動画1本ぶんの特徴（コマの平均）。画像の特徴とは別のキーで保存する


def model_path(cfg, kind="fp32", log=print) -> Path:
    return bench.ensure_model(cfg.app_root / "models" / MODEL_NAME, kind, log)


def pending_assets(conn, mounts, limit=None, retry=False, model=None):
    """まだ特徴ベクトルのない実写画像アセットと、読むファイルのパスを返す。[(asset_id, path)]"""
    from .report import abs_path

    rows = conn.execute(
        """SELECT a.id AS aid, f.vol_id, f.volpath, f.ext, (a.master_file_id = f.id) AS is_master
           FROM assets a JOIN files f ON f.asset_id = a.id AND f.status='present'
           LEFT JOIN embeddings e ON e.asset_id = a.id AND e.model = ?
           WHERE a.kind='image' AND a.zone='real' AND (e.asset_id IS NULL"""
        + (" OR e.vec IS NULL" if retry else "") + ")",
        (model or MODEL_NAME,),
    ).fetchall()
    best: dict = {}
    for r in rows:
        if "." + str(r["ext"]).lower().lstrip(".") not in bench.IMAGE_EXTS:
            continue
        cur = best.get(r["aid"])
        if cur is None or (r["is_master"] and not cur[1]):
            best[r["aid"]] = (r, r["is_master"])
    out = []
    for aid in sorted(best):
        r = best[aid][0]
        p = abs_path(mounts, r["vol_id"], r["volpath"])
        if os.path.exists(p):
            out.append((aid, p))
    return out[:limit] if limit else out


def run(cfg, conn, kind="fp32", model=None, limit=None, retry=False, batch=8, chunk=64, log=print) -> int:
    import numpy as np
    import onnxruntime as ort

    from .report import _mounts

    try:
        mp = Path(model) if model else model_path(cfg, kind, log)
    except Exception as ex:
        log(f"モデルを取得できませんでした: {ex}")
        return 3
    todo = pending_assets(conn, _mounts(conn), limit, retry)
    done_before = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model=? AND vec IS NOT NULL", (MODEL_NAME,)).fetchone()[0]
    log(f"特徴ベクトルを付ける画像: {len(todo):,} 件（済み {done_before:,} 件）")
    if not todo:
        return 0
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(mp), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    cpus = os.cpu_count() or 1

    def prep(p):
        try:
            return bench.preprocess(p)
        except Exception as ex:
            return ex

    t0, n_ok, n_bad = time.time(), 0, 0
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with ThreadPoolExecutor(max_workers=max(1, cpus - 2)) as ex:
            for i in range(0, len(todo), chunk):
                part = todo[i:i + chunk]
                res = list(ex.map(prep, [p for _, p in part]))
                ok = [(aid, a) for (aid, _), a in zip(part, res) if not isinstance(a, Exception)]
                rows = [(aid, MODEL_NAME, 0, None, f"{type(a).__name__}: {a}"[:200], stamp)
                        for (aid, _), a in zip(part, res) if isinstance(a, Exception)]
                n_bad += len(rows)
                embs = []
                for j in range(0, len(ok), batch):
                    arr = np.stack([a for _, a in ok[j:j + batch]])
                    o = sess.run(outs, {inp: arr})
                    e = bench._pick_embedding(o)
                    embs.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12))
                if embs:
                    E = np.concatenate(embs).astype(np.float16)
                    for (aid, _), v in zip(ok, E):
                        rows.append((aid, MODEL_NAME, int(v.shape[0]), v.tobytes(), None, stamp))
                    n_ok += len(ok)
                conn.executemany("INSERT OR REPLACE INTO embeddings(asset_id,model,dim,vec,err,created) VALUES(?,?,?,?,?,?)", rows)
                conn.commit()
                el = time.time() - t0
                k = i + len(part)
                eta = el / k * (len(todo) - k)
                log(f"  {k:,}/{len(todo):,}  {k / el:5.1f}枚/秒  残り約{bench.fmt_time(eta)}  (失敗 {n_bad})")
    except KeyboardInterrupt:
        conn.commit()
        log("中断しました。もう一度実行すると続きから再開します。")
        return 1
    log(f"完了: 成功 {n_ok:,} 件 / 失敗 {n_bad:,} 件 / {bench.fmt_time(time.time() - t0)}")
    return 0


def load_matrix(conn, ids=None, with_video=False, model=None):
    """(asset_id の配列, float32 の行列) を返す。with_video=True なら、動画（コマの平均）も加える（CLIP のときだけ）。
    model を指定すると、別のモデル（'dinov2-small'）の特徴を返す。"""
    import numpy as np

    model = model or MODEL_NAME
    models = (MODEL_NAME, VIDEO_MODEL) if (with_video and model == MODEL_NAME) else (model,)
    rows = conn.execute(f"SELECT asset_id, dim, vec FROM embeddings WHERE model IN ({','.join('?' * len(models))}) AND vec IS NOT NULL", models).fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    ids_ = np.array([r["asset_id"] for r in rows], dtype=np.int64)
    X = np.stack([np.frombuffer(r["vec"], dtype=np.float16) for r in rows]).astype(np.float32)
    return ids_, X


DINO_MODEL = "dinov2-small"


def pet_model(conn) -> str:
    """ペットの個体判定に使う特徴のモデル。DINOv2 が（CLIP の画像のほぼ全部に）付いていれば DINOv2、なければ CLIP。
    meta の 'pet_model' に 'clip' か 'dinov2-small' を入れると、それを使う。"""
    r = conn.execute("SELECT value FROM meta WHERE key='pet_model'").fetchone()
    if r and r[0] in (MODEL_NAME, "clip", DINO_MODEL):
        return MODEL_NAME if r[0] in (MODEL_NAME, "clip") else DINO_MODEL
    n_d = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model=? AND vec IS NOT NULL", (DINO_MODEL,)).fetchone()[0]
    n_c = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model=? AND vec IS NOT NULL", (MODEL_NAME,)).fetchone()[0]
    return DINO_MODEL if n_c and n_d >= 0.95 * n_c else MODEL_NAME
