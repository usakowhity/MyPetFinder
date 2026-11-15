"""AI分析（Step 2）の速度測定。画像の読み込みと、CLIP の画像エンコーダの速度を、使える機器ごとに測る。

使い方: python mypet.py bench   （先に setup_ai.bat で必要なライブラリを入れておく）
測るもの
  ① 画像の読み込み・縮小（CPU）           … 単独 / 複数並列
  ② CLIP 画像エンコーダ（ViT-B/32）      … onnxruntime(CPU) と OpenVINO の CPU / GPU(内蔵) / NPU
結果は画面と data/reports/bench_ai.txt に出す。元の写真・動画は読むだけで、何も変更しない。
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_FILES = {
    "fp32": "vision_model.onnx",           # 352MB
    "int8": "vision_model_quantized.onnx",  # 89MB（速いが精度は少し落ちる）
}
MODEL_URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/{name}"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------- 画像の準備
def pick_samples(conn, mounts: dict, n: int, seed: int = 0, log=print):
    """目録から、実写の画像を n 枚ランダムに選ぶ（存在するものだけ）。(パスの一覧, 実写画像の総数) を返す。"""
    from .report import abs_path

    rows = [
        r for r in conn.execute(
            "SELECT vol_id, volpath, ext FROM files WHERE status='present' AND zone='real' AND kind='image'"
        )
        if "." + str(r["ext"]).lower().lstrip(".") in IMAGE_EXTS
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out, missing = [], 0
    for r in rows:
        p = abs_path(mounts, r["vol_id"], r["volpath"])
        if os.path.exists(p):
            out.append(p)
            if len(out) >= n:
                break
        else:
            missing += 1
    if missing:
        log(f"  注意: 目録にあるのに見つからないファイルが {missing} 件ありました（ドライブが接続されていない可能性）。")
    return out, len(rows)


def _prep_pil(im, mean=None, std=None, resize=None, crop=None):
    """PIL画像（RGB）→ モデルの入力。float32 の (3,crop,crop)。既定は CLIP（短い辺を224に縮小 → 中央224角 → 正規化）。
    resize/crop/mean/std を変えると、別のモデル（DINOv2: 256→224、ImageNet の平均・標準偏差）にも使える。"""
    import numpy as np
    from PIL import Image

    crop = crop or SIZE
    resize = resize or crop
    mean = MEAN if mean is None else mean
    std = STD if std is None else std
    w, h = im.size
    s = resize / min(w, h)
    nw, nh = max(crop, round(w * s)), max(crop, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - crop) // 2, (nh - crop) // 2
    im = im.crop((l, t, l + crop, t + crop))
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return a.transpose(2, 0, 1).copy()


def preprocess(path: str):
    """CLIP の前処理: 短い辺を224に縮小 → 中央を224角に切り出し → 平均・標準偏差で正規化。float32 の (3,224,224)。"""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.draft("RGB", (SIZE * 2, SIZE * 2))   # JPEGは、縮小して読み込める（速い）。それ以外では何もしない
        im = ImageOps.exif_transpose(im).convert("RGB")
        return _prep_pil(im)


def preprocess_bgr(arr):
    """OpenCV の BGR 配列（動画のコマ）→ CLIP の入力。"""
    import numpy as np
    from PIL import Image

    return _prep_pil(Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1])))


def load_all(paths, workers: int):
    """paths をすべて前処理して返す。(配列の一覧, 失敗したパス, 秒)"""
    t0 = time.perf_counter()
    arrays, bad = [], []

    def one(p):
        try:
            return preprocess(p)
        except Exception:
            return None

    if workers <= 1:
        res = [one(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(one, paths))
    for p, a in zip(paths, res):
        (bad.append(p) if a is None else arrays.append(a))
    return arrays, bad, time.perf_counter() - t0


# ---------------------------------------------------------------- モデルの取得
def download(url: str, dest: Path, log=print) -> None:
    """途中から再開できるダウンロード。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "MyPetDatabase-bench"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        status = getattr(resp, "status", 200)
        mode = "ab" if (have and status == 206) else "wb"
        total = resp.headers.get("Content-Length")
        total = int(total) + (have if mode == "ab" else 0) if total else None
        done = have if mode == "ab" else 0
        last = 0.0
        with open(part, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 2:
                    last = time.time()
                    pct = f" ({done * 100 // total}%)" if total else ""
                    log(f"    ダウンロード中... {done / 1e6:,.0f} MB{pct}")
    os.replace(part, dest)


def ensure_model(model_dir: Path, kind: str, log=print) -> Path:
    name = MODEL_FILES[kind]
    path = model_dir / name
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    url = MODEL_URL.format(name=name)
    log(f"  モデルを取得します: {url}")
    download(url, path, log)
    return path


# ---------------------------------------------------------------- 測定
def _pick_embedding(outputs):
    """出力のうち、(N, 次元) の2次元のもの（画像の特徴ベクトル）を選ぶ。"""
    for o in outputs:
        if getattr(o, "ndim", 0) == 2:
            return o
    return outputs[0]


def _batches(arrays, b):
    import numpy as np

    n = len(arrays) - len(arrays) % b if b > 1 else len(arrays)   # 端数は切り捨て（静的な形のモデル用）
    return [np.stack(arrays[i:i + b]) for i in range(0, n, b)], n


def run_ort(model_path: Path, arrays, provider: str, batch: int):
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=[provider])
    inp = sess.get_inputs()[0]
    outs = [o.name for o in sess.get_outputs()]
    batches, n = _batches(arrays, batch)
    sess.run(outs, {inp.name: batches[0]})            # ウォームアップ
    emb, t0 = [], time.perf_counter()
    for b in batches:
        emb.append(_pick_embedding(sess.run(outs, {inp.name: b})))
    dt = time.perf_counter() - t0
    return n / dt, np.concatenate(emb), n


def run_ov(model_path: Path, arrays, device: str, batch: int):
    import numpy as np
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(model_path))
    name = model.inputs[0].get_any_name()
    model.reshape({name: [batch, 3, SIZE, SIZE]})      # NPU は形が固定でないと動かない
    cfg = {} if device == "NPU" else {"PERFORMANCE_HINT": "THROUGHPUT"}
    comp = core.compile_model(model, device, cfg)
    out_idx = 0
    for i, o in enumerate(comp.outputs):
        if len(o.get_partial_shape()) == 2:
            out_idx = i
            break
    batches, n = _batches(arrays, batch)
    comp(batches[0])                                   # ウォームアップ
    try:
        jobs = max(1, int(comp.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
    except Exception:
        jobs = 1
    results: dict = {}
    q = ov.AsyncInferQueue(comp, jobs)
    q.set_callback(lambda req, i: results.__setitem__(i, req.get_output_tensor(out_idx).data.copy()))
    t0 = time.perf_counter()
    for i, b in enumerate(batches):
        q.start_async({0: b}, userdata=i)
    q.wait_all()
    dt = time.perf_counter() - t0
    emb = np.concatenate([results[i] for i in range(len(batches))])
    return n / dt, emb, n


def cosine_min(a, b, k=16):
    import numpy as np

    k = min(k, len(a), len(b))
    x, y = a[:k], b[:k]
    c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9)
    return float(c.min())


def fmt_time(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f}秒"
    if sec < 5400:
        return f"{sec / 60:.0f}分"
    return f"{sec / 3600:.1f}時間"


def run(cfg, conn, n=200, kind="fp32", model=None, devices=None, log=print) -> int:
    from .report import _mounts

    lines: list[str] = []

    def out(s=""):
        lines.append(s)
        log(s)

    out(f"=== AI分析 速度測定  ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
    try:
        import numpy as np  # noqa: F401
        import PIL

        out(f"Python {sys.version.split()[0]} / Pillow {PIL.__version__}")
    except ImportError as ex:
        out(f"必要なライブラリが入っていません: {ex}")
        out("先に setup_ai.bat を実行してください。")
        return 2
    cpus = os.cpu_count() or 1
    out(f"CPU論理コア数: {cpus}")

    ort = ov = None
    try:
        import onnxruntime as ort

        out(f"onnxruntime {ort.__version__}  利用できる実行先: {', '.join(ort.get_available_providers())}")
    except ImportError:
        out("onnxruntime: 入っていません（スキップ）")
    try:
        import openvino as ov

        core = ov.Core()
        names = []
        for d in core.available_devices:
            try:
                names.append(f"{d}={core.get_property(d, 'FULL_DEVICE_NAME')}")
            except Exception:
                names.append(d)
        out(f"OpenVINO {ov.__version__}  利用できる機器: {' / '.join(names)}")
    except ImportError:
        out("OpenVINO: 入っていません（スキップ）")
    if ort is None and ov is None:
        out("onnxruntime も OpenVINO も入っていないので、測定できません。setup_ai.bat を実行してください。")
        return 2

    # モデル
    try:
        model_path = Path(model) if model else ensure_model(cfg.app_root / "models" / "clip-vit-b32", kind, out)
    except Exception as ex:
        out(f"モデルを取得できませんでした: {ex}")
        out("ネットワークの制限が原因かもしれません。次のファイルを手で保存して、--model でパスを指定してもできます:")
        out("  " + MODEL_URL.format(name=MODEL_FILES[kind]))
        return 3
    out(f"モデル: {model_path}  ({model_path.stat().st_size / 1e6:,.0f} MB, {kind})")

    # 画像
    paths, total_images = pick_samples(conn, _mounts(conn), n, log=out)
    if not paths:
        out("目録に、読める実写の画像がありません。先に Scan を実行してください。")
        return 4
    out(f"\n[画像の読み込み・縮小]  対象 {len(paths)} 枚（実写の画像は目録に全部で {total_images:,} 枚）")
    a1, bad, t1 = load_all(paths, 1)
    out(f"  1スレッド : {len(a1) / t1:6.1f} 枚/秒")
    aN, bad, tN = load_all(paths, cpus)
    out(f"  {cpus}スレッド : {len(aN) / tN:6.1f} 枚/秒")
    if bad:
        out(f"  読めなかった画像: {len(bad)} 枚（測定からは除きました）")
    arrays = aN
    if len(arrays) < 8:
        out("測定できる画像が少なすぎます。")
        return 4
    prep_ips = len(arrays) / tN

    # 推論
    want = devices or ["CPU", "GPU", "NPU"]
    plan = []
    if ort is not None:
        plan.append(("onnxruntime CPU", lambda b: run_ort(model_path, arrays, "CPUExecutionProvider", b), (1, 8)))
        for p in ort.get_available_providers():
            if p in ("DmlExecutionProvider", "OpenVINOExecutionProvider", "CUDAExecutionProvider"):
                plan.append((f"onnxruntime {p.replace('ExecutionProvider', '')}",
                             (lambda p_: lambda b: run_ort(model_path, arrays, p_, b))(p), (1, 8)))
    if ov is not None:
        have = set(ov.Core().available_devices)
        for d in want:
            if d in have or any(x.startswith(d) for x in have):
                plan.append((f"OpenVINO {d}", (lambda d_: lambda b: run_ov(model_path, arrays, d_, b))(d),
                             (1,) if d == "NPU" else (1, 8)))
            else:
                out(f"  （OpenVINO {d}: この機器は見つかりませんでした）")

    out(f"\n[CLIP 画像エンコーダ ({kind})]  {len(arrays)} 枚")
    results, ref = [], None
    for label, fn, batches in plan:
        for b in batches:
            tag = f"{label} (バッチ{b})"
            try:
                ips, emb, used = fn(b)
            except Exception as ex:
                out(f"  {tag:<34} 失敗: {type(ex).__name__}: {str(ex).splitlines()[0][:110] if str(ex) else ''}")
                continue
            if ref is None and label == "onnxruntime CPU" and b == 1:
                ref = emb
            note = ""
            if ref is not None and not (label == "onnxruntime CPU" and b == 1):
                note = f"  参照との一致(cos最小) {cosine_min(ref, emb):.4f}"
            out(f"  {tag:<34} {ips:7.1f} 枚/秒{note}")
            results.append((tag, ips))
    if not results:
        out("測定できた機器がありません。")
        return 5

    best_tag, best_ips = max(results, key=lambda x: x[1])
    pipeline = min(best_ips, prep_ips)   # 読み込みと推論が重ならない場合は合算、重なれば遅いほう。安全側に合算で見る
    serial = 1 / (1 / best_ips + 1 / prep_ips)
    out("\n[見積り]")
    out(f"  最速: {best_tag}  {best_ips:.1f} 枚/秒 / 画像の読み込み {prep_ips:.1f} 枚/秒")
    out(f"  実写の画像 {total_images:,} 枚: 読み込みと推論を並行できれば 約{fmt_time(total_images / pipeline)}、"
        f"順番に行えば 約{fmt_time(total_images / serial)}")
    out("  ※ 動画（実写 約3,600本）は、1本から数枚のコマを取り出す必要があり、別に見積もります。")
    out("  ※ 200枚の測定は、起動直後の一時的な速さのぶれを含みます。ほかのアプリを閉じて測ると安定します。")

    rep = cfg.db_path.parent / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "bench_ai.txt").write_text("\n".join(lines), encoding="utf-8")
    out(f"\n結果を保存しました: {rep / 'bench_ai.txt'}")
    return 0
