"""CLIP のゼロショット判定: 「生きたペット」か「額入り写真・画面・置物・印刷物・ペットなし」かを、言葉との近さで決める。

学習は使わない。画像の特徴ベクトル（Embed 済み）と、文章の特徴ベクトルのコサイン類似度で、プロンプトの確率（softmax）を出す。
 ・文章側のモデル（text_model.onnx、約250MB）と、語彙（vocab.json / merges.txt）を初回だけ取得する
 ・点数: p_live = 「生きたペット」のプロンプトの確率の合計（1 - p_live が「いない」らしさ）
 ・結果は asset_zeroshot に保存し、登録フォルダ（nopet_folders.toml）などで当たり方を表示する。Identify にはまだ使わない
"""
from __future__ import annotations

import json
import random
import re
import time
from functools import lru_cache
from pathlib import Path

from . import bench, embed, nopetfolders

BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
FILES = {"text_model.onnx": BASE + "onnx/text_model.onnx", "vocab.json": BASE + "vocab.json", "merges.txt": BASE + "merges.txt"}
BOS, EOS, CTX = 49406, 49407, 77
LOGIT_SCALE = 100.0

# クラス名 → プロンプト。live が「生きたペット」
PROMPTS = {
    "live": ["a photo of a pet rabbit", "a photo of a rabbit sitting on the floor", "a close-up photo of a rabbit", "a photo of a pet rat"],
    "framed": ["a photo of a framed portrait photo of a rabbit", "a photo of a framed picture on a shelf", "a memorial photo of a pet with flowers"],
    "screen": ["a photo of a television screen showing a rabbit", "a photo of a computer monitor screen", "a photo of a smartphone screen"],
    "figurine": ["a photo of a rabbit figurine", "a photo of a stuffed toy rabbit", "a photo of a rabbit shaped decoration"],
    "print": ["a photo of a catalog page with photos of rabbits", "a photo of a printed brochure", "a photo of a cushion with a rabbit pattern"],
    "nopet": ["a photo of a person", "a photo of a room", "a photo of food", "a photo of a landscape"],
}
LABEL = {"live": "生きたペット", "framed": "額入り写真", "screen": "画面", "figurine": "置物・縫いぐるみ", "print": "印刷物・絵柄", "nopet": "ペットなし"}


# ---------------------------------------------------------------- トークナイザ（CLIP の BPE。英語のプロンプト専用）
def _bytes_to_unicode() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_PAT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+")


class Tokenizer:
    def __init__(self, vocab: dict, merges: list):
        self.enc = vocab
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.b2u = _bytes_to_unicode()
        self.cache: dict = {}

    @classmethod
    def load(cls, d: Path):
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        lines = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
        merges = [m for m in lines[1:] if m.strip()]
        return cls(vocab, merges)

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            a, b = best
            new, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list:
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = [BOS]
        for tok in _PAT.findall(text):
            u = "".join(self.b2u[b] for b in tok.encode("utf-8"))
            ids += [self.enc[t] for t in self._bpe(u).split(" ")]
        ids = ids[: CTX - 1] + [EOS]
        return ids

    def pad(self, ids: list) -> tuple:
        n = len(ids)
        return ids + [EOS] * (CTX - n), [1] * n + [0] * (CTX - n)


def selfcheck(tok: Tokenizer) -> bool:
    """既知の正解（OpenAI の CLIP）と一致するか。"""
    return tok.encode("a photo of a cat") == [49406, 320, 1125, 539, 320, 2368, 49407]


# ---------------------------------------------------------------- 文章の特徴ベクトル
def ensure_files(model_dir: Path, log=print) -> None:
    for name, url in FILES.items():
        p = model_dir / name
        if p.exists() and p.stat().st_size > 1000:
            continue
        log(f"  取得します: {url}")
        bench.download(url, p, log)


_CACHE: dict = {}


def _load(model_dir: Path):
    """トークナイザと文章モデルを、1回だけ読み込んで使い回す（モデルの読み込みに数秒かかるため）。"""
    import onnxruntime as ort

    key = str(model_dir)
    if key not in _CACHE:
        tok = Tokenizer.load(model_dir)
        if not selfcheck(tok):
            raise RuntimeError("トークナイザの自己確認に失敗しました（vocab.json / merges.txt を確認してください）")
        sess = ort.InferenceSession(str(model_dir / "text_model.onnx"), providers=["CPUExecutionProvider"])
        _CACHE[key] = (tok, sess)
    return _CACHE[key]


def text_embeddings(model_dir: Path, prompts: list, log=print):
    import numpy as np

    tok, sess = _load(model_dir)
    ids, mask = zip(*(tok.pad(tok.encode(p)) for p in prompts))
    names = {i.name for i in sess.get_inputs()}
    feed = {"input_ids": np.array(ids, dtype=np.int64)}
    if "attention_mask" in names:
        feed["attention_mask"] = np.array(mask, dtype=np.int64)
    outs = sess.run(None, feed)
    emb = next((o for o in outs if getattr(o, "ndim", 0) == 2), None)
    if emb is None:
        raise RuntimeError("文章モデルの出力の形が想定と違います")
    emb = emb.astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def features(model_dir: Path, X):
    """画像の特徴 X に付け足す、ゼロショットの特徴（各クラスの確率の対数）。文章の特徴が未保存なら None。"""
    import numpy as np

    p = model_dir / "zs_text.npy"
    if not p.exists():
        return None
    T = np.load(p)
    groups = [c for c, ps in PROMPTS.items() for _ in ps]
    if T.shape[0] != len(groups):
        return None
    _, P = classify(X, T, groups)
    return (np.log(P + 1e-3) / 2.0).astype(np.float32)


# ---------------------------------------------------------------- 実行
def classify(X, T, groups: list):
    """X: (N, D) 画像（正規化済み）、T: (P, D) 文章。groups: プロンプトごとのクラス名。(クラス名の一覧, (N, クラス数) の確率)"""
    import numpy as np

    logits = LOGIT_SCALE * (X.astype(np.float32) @ T.T)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(axis=1, keepdims=True)
    classes = list(dict.fromkeys(groups))
    cols = [p[:, [i for i, g in enumerate(groups) if g == c]].sum(axis=1) for c in classes]
    return classes, np.stack(cols, axis=1)


def run(cfg, conn, log=print, seed=0) -> int:
    import numpy as np

    ids, X = embed.load_matrix(conn)
    if len(ids) == 0:
        log("特徴ベクトルがありません。先にメニュー8（Embed）を実行してください。")
        return 2
    model_dir = cfg.app_root / "models" / embed.MODEL_NAME
    try:
        ensure_files(model_dir, log)
        groups = [c for c, ps in PROMPTS.items() for _ in ps]
        prompts = [p for ps in PROMPTS.values() for p in ps]
        T = text_embeddings(model_dir, prompts, log)
    except Exception as ex:                       # 通信・モデルの問題は、わかる言葉で伝える
        log(f"文章モデルの準備に失敗しました: {type(ex).__name__}: {ex}")
        return 3
    try:
        np.save(model_dir / "zs_text.npy", T)          # NoPet が、ゼロショットの点数を特徴として併用する
    except OSError:
        pass
    classes, P = classify(X, T, groups)
    live = classes.index("live")
    s = 1.0 - P[:, live]                          # 「いない」らしさ
    top = [classes[i] for i in P.argmax(axis=1)]
    pos = {int(a): i for i, a in enumerate(ids)}
    stamp = time.strftime("%Y-%m-%d")
    conn.execute("CREATE TABLE IF NOT EXISTS asset_zeroshot(asset_id INTEGER PRIMARY KEY, p_live REAL, top TEXT, created TEXT)")
    conn.execute("DELETE FROM asset_zeroshot")
    conn.executemany("INSERT INTO asset_zeroshot(asset_id,p_live,top,created) VALUES(?,?,?,?)",
                     [(int(a), float(P[i, live]), top[i], stamp) for i, a in enumerate(ids)])
    conn.commit()

    def rates(label, assets, want_high=True):
        a = [pos[x] for x in assets if x in pos]
        if not a:
            return
        v = s[a]
        log(f"  {label:<26}" + " / ".join(f"{t:.1f}以上 {float((v >= t).mean()) * 100:5.1f}%" for t in (0.5, 0.7, 0.9)) + f"  （{len(a):,} 件）")

    by_kind, _ = nopetfolders.assets_by_kind(conn, nopetfolders.load(cfg))
    log("\n【ゼロショット】「生きたペットではない」と判定される率（しきい値は「いない」らしさ）")
    log("◆ 生きたペットが写っていない写真（高いほどよい）")
    kinds = {}
    for (kind, lv), a in by_kind.items():
        if not lv:
            kinds.setdefault(kind, set()).update(a)
    for kind, a in sorted(kinds.items()):
        rates(f"登録: {kind}", a)
    none_rev = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM photo_reviews WHERE answer='none' AND asset_id IS NOT NULL")}
    rates("確認済み「いない」", none_rev)
    log("◆ 生きた動物が写っている写真（低いほどよい）")
    other = set().union(*[a for (k, lv), a in by_kind.items() if lv]) if any(lv for _, lv in by_kind) else set()
    rates("他のウサギ", other)
    live_rev = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM photo_reviews WHERE answer<>'none' AND asset_id IS NOT NULL")}
    rates("確認済み「ペット名」", live_rev)
    from .petcheck import collect_labels
    lab = [a for a in collect_labels(conn) if a in pos and a not in other]
    random.Random(seed).shuffle(lab)
    rates("名前の推定の写真（無作為）", lab[:3000])
    log("  ※ 名前の推定の写真には、実際はペットがいない写真も含まれるため、実際の誤りはこれより少ない")
    log("\n◆ 判定された種類の内訳（登録フォルダ）")
    for kind, a in sorted(kinds.items()):
        c = {}
        for x in a:
            if x in pos:
                c[top[pos[x]]] = c.get(top[pos[x]], 0) + 1
        log(f"  {kind}: " + ", ".join(f"{LABEL[k]} {n}" for k, n in sorted(c.items(), key=lambda kv: -kv[1])))
    log(f"\n全画像の判定の内訳: " + ", ".join(f"{LABEL[c]} {int((np.array(top) == c).sum()):,}" for c in classes))
    return 0
