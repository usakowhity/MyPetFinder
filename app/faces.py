"""顔の検出と、顔ごとの特徴の保存（人物の判定の準備）。OpenCV 付属の2つのモデルを使う。

 ・検出: YuNet（MIT）  ・特徴: SFace（Apache-2.0）。どちらも ONNX で、CPU で動く。取得元は Hugging Face の opencv 組織。
 ・実写の画像だけが対象。元の写真は読むだけで、変更しない。結果は faces / face_scan に保存する（途中から再開できる）
 ・顔のデータは個人データなので、このPCの中だけに置く（配布版には含めない）
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import bench

HF = "https://huggingface.co/opencv/"
MODELS = {
    "face_detection_yunet_2023mar.onnx": HF + "face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": HF + "face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
}
MAXSIDE = 1280          # 検出前に、長い辺をここまでに縮める
MIN_FACE = 28           # 縮めた画像で、これより小さい顔は使わない（ピクセル）
MIN_SCORE = 0.85


def model_dir(cfg) -> Path:
    return cfg.app_root / "models" / "faces"


def ensure_models(d: Path, log=print) -> None:
    for name, url in MODELS.items():
        p = d / name
        if p.exists() and p.stat().st_size > 100_000:
            continue
        log(f"  取得します: {url}")
        bench.download(url, p, log)


class Engine:
    """OpenCV の顔検出（YuNet）と顔の特徴（SFace）。detect(bgr) → [(x, y, w, h, score, 特徴(128,))]（座標は画素）"""

    def __init__(self, d: Path):
        import cv2

        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(str(d / "face_detection_yunet_2023mar.onnx"), "", (320, 320), MIN_SCORE, 0.3, 2000)
        self.rec = cv2.FaceRecognizerSF.create(str(d / "face_recognition_sface_2021dec.onnx"), "")

    def detect(self, bgr):
        import numpy as np

        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        out = []
        if faces is None:
            return out
        for row in faces:
            x, y, fw, fh, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[14])
            if score < MIN_SCORE or min(fw, fh) < MIN_FACE:
                continue
            aligned = self.rec.alignCrop(bgr, row)
            f = np.asarray(self.rec.feature(aligned), dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(f))
            if n > 0:
                out.append((x, y, fw, fh, score, f / n))
        return out


def load_bgr(path):
    """画像を読み、向きを直し、長い辺を MAXSIDE までに縮めて BGR の配列にする。"""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAXSIDE, MAXSIDE))
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def pending(conn, limit=None, retry=False):
    from .report import _mounts, abs_path

    mounts = _mounts(conn)
    done = "" if retry else " AND a.id NOT IN (SELECT asset_id FROM face_scan)"
    if retry:
        done = " AND a.id NOT IN (SELECT asset_id FROM face_scan WHERE err IS NULL)"
    rows = conn.execute(
        f"""SELECT a.id, MIN(f.id) fid FROM assets a JOIN files f ON f.asset_id=a.id AND f.status='present'
            WHERE a.zone='real' AND a.kind='image'{done} GROUP BY a.id ORDER BY a.id""").fetchall()
    out = []
    for r in rows[: limit or None]:
        f = conn.execute("SELECT vol_id, volpath FROM files WHERE id=?", (r["fid"],)).fetchone()
        out.append((r["id"], abs_path(mounts, f["vol_id"], f["volpath"])))
    return out


def run(cfg, conn, limit=None, retry=False, engine=None, chunk=64, log=print) -> int:
    import numpy as np

    if engine is None:
        d = model_dir(cfg)
        try:
            ensure_models(d, log)
            engine = Engine(d)
        except Exception as ex:
            log(f"顔のモデルの準備に失敗しました: {type(ex).__name__}: {ex}\n  setup_ai.bat を実行して opencv-python-headless を入れたか、通信できるかを確認してください。")
            return 3
    todo = pending(conn, limit, retry)
    done_before = conn.execute("SELECT COUNT(*) FROM face_scan WHERE err IS NULL").fetchone()[0]
    log(f"顔を探す画像: {len(todo):,} 件（済み {done_before:,} 件）")
    if not todo:
        return 0
    t0 = time.time()
    n_img = n_face = n_err = 0
    stamp = time.strftime("%Y-%m-%d")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for s in range(0, len(todo), chunk):
            part = todo[s: s + chunk]

            def load(item):
                try:
                    return load_bgr(item[1]), None
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"[:200]
            for (aid, _), (img, err) in zip(part, ex.map(load, part)):
                if retry:
                    conn.execute("DELETE FROM faces WHERE asset_id=?", (aid,))
                if img is None:
                    conn.execute("INSERT OR REPLACE INTO face_scan(asset_id,n,err,created) VALUES(?,0,?,?)", (aid, err, stamp))
                    n_err += 1
                    continue
                h, w = img.shape[:2]
                try:
                    found = engine.detect(img)
                except Exception as e:
                    conn.execute("INSERT OR REPLACE INTO face_scan(asset_id,n,err,created) VALUES(?,0,?,?)", (aid, f"{type(e).__name__}: {e}"[:200], stamp))
                    n_err += 1
                    continue
                for (x, y, fw, fh, score, f) in found:
                    conn.execute("INSERT INTO faces(asset_id,x,y,w,h,score,vec,created) VALUES(?,?,?,?,?,?,?,?)",
                                 (aid, x / w, y / h, fw / w, fh / h, score, f.astype(np.float16).tobytes(), stamp))
                conn.execute("INSERT OR REPLACE INTO face_scan(asset_id,n,err,created) VALUES(?,?,NULL,?)", (aid, len(found), stamp))
                n_img += 1
                n_face += len(found)
            conn.commit()
            el = time.time() - t0
            k = min(s + chunk, len(todo))
            log(f"  {k:,}/{len(todo):,}  {k / el:5.1f}枚/秒  残り約{(len(todo) - k) / max(k / el, 1e-9) / 60:.0f}分  顔 {n_face:,} 件")
    log(f"完了: 画像 {n_img:,} 枚 / 顔 {n_face:,} 件 / 失敗 {n_err:,} 件 / {time.time() - t0:.0f}秒")
    return 0
