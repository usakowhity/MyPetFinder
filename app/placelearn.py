"""場所を、人が付けた少数の写真と見た目（CLIP）が似ているかで、ほかの写真にも広げる。

 ・人が付けた写真（place_labels, positive=1）との類似度 = 上位3枚のコサイン類似度の平均
 ・「似た写真を探す」: まだ付いていない写真を、類似度の高い順に並べる（確認して一括で付けられる）
 ・「自動で付ける」: 類似度がしきい値以上の写真を place_ai に入れる（検索では「AI推定」として使う）
 ・「この場所ではない」(positive=0) と答えた写真は、自動では付けない
"""
from __future__ import annotations

import time

K = 3
DEFAULT_THR = 0.92          # 場所・場面の自動判定の初期値（実際の写真・動画で調整した値）
# 種類ごとの（人が付けた表, AI推定の表, 列名, meta のキー）
KINDS = {"place": ("place_labels", "place_ai", "place_id", "place_thr"), "tag": ("tag_labels", "tag_ai", "tag_id", "tag_thr")}


def _t(kind):
    if kind not in KINDS:
        raise ValueError("kind")
    return KINDS[kind]


def _labels(conn, pid, positive, kind="place"):
    lt, _, col, _ = _t(kind)
    return [r[0] for r in conn.execute(f"SELECT asset_id FROM {lt} WHERE {col}=? AND positive=?", (pid, positive))]


def _topk_mean(S):
    import numpy as np

    k = min(K, S.shape[1])
    top = np.partition(S, S.shape[1] - k, axis=1)[:, -k:] if S.shape[1] > k else S
    return top.mean(axis=1)


def scores(conn, pos, X, pid, kind="place"):
    """全アセットの類似度（上位K枚の平均）。見本（人が付けた写真）がなければ None。(行列の順は X に同じ)
    動画は、コマごとの類似度の平均にする（コマの平均の特徴より、写真と同じ尺度で比べられるため）。"""
    import numpy as np

    ex = [pos[a] for a in _labels(conn, pid, 1, kind) if a in pos]
    if not ex:
        return None
    out = _topk_mean(X @ X[ex].T)                     # (N, 見本の数) → 上位K枚の平均
    out[ex] = 1.0                                     # 見本自身
    from . import video

    fa, FX = video.load_frames_cached(conn)
    if len(fa):
        fs = _topk_mean(FX @ X[ex].T)
        order = np.unique(fa, return_index=True)
        starts = order[1]
        means = np.add.reduceat(fs, starts) / np.diff(np.append(starts, len(fa)))
        for a, m in zip(order[0], means):
            i = pos.get(int(a))
            if i is not None and i not in ex:
                out[i] = m
    return out


def _real_images(conn):
    return {r[0] for r in conn.execute("SELECT id FROM assets WHERE zone='real' AND kind IN ('image','video')")}


def _excluded(conn, pid, kind="place"):
    return set(_labels(conn, pid, 1, kind)) | set(_labels(conn, pid, 0, kind))


def suggest(conn, pos, X, ids_by_row, pid, limit=60, offset=0, kind="place"):
    """まだ付いていない写真を、類似度の高い順に。(asset_id, sim) の一覧と、候補の総数。"""
    import numpy as np

    sc = scores(conn, pos, X, pid, kind)
    if sc is None:
        return [], 0
    ok = _real_images(conn) - _excluded(conn, pid, kind)
    order = np.argsort(-sc)
    cand = [(int(ids_by_row[i]), float(sc[i])) for i in order if int(ids_by_row[i]) in ok]
    return cand[offset: offset + limit], len(cand)


def count_at(conn, pos, X, ids_by_row, pid, thr, kind="place"):
    sc = scores(conn, pos, X, pid, kind)
    if sc is None:
        return 0
    ok = _real_images(conn) - _excluded(conn, pid, kind)
    return sum(1 for i in range(len(sc)) if sc[i] >= thr and int(ids_by_row[i]) in ok)


def loo_recall(conn, pos, X, pid, thr, kind="place"):
    """見本1枚ずつを外して、残りとの類似度がしきい値以上になる割合（見本が3枚以上のとき）。"""
    import numpy as np

    ex = [pos[a] for a in _labels(conn, pid, 1, kind) if a in pos]
    if len(ex) < 3:
        return None
    S = X[ex] @ X[ex].T
    np.fill_diagonal(S, -1)
    k = min(K, len(ex) - 1)
    top = np.sort(S, axis=1)[:, -k:].mean(axis=1)
    return float((top >= thr).mean())


def user_key(kind, pid) -> str:
    return f"{_t(kind)[3]}_user:{pid}"


def applied_by_user(conn, pid, kind="place") -> bool:
    """人が「自動で付ける」を押した場所・場面か（押していないものには、自動の判定を勝手に付けない）。"""
    return conn.execute("SELECT 1 FROM meta WHERE key=?", (user_key(kind, pid),)).fetchone() is not None


def clear(conn, pid, kind="place") -> int:
    """自動の判定（AI推定）を消す。人が付けた見本は残す。しきい値と「自動で付けた」記録も消す。"""
    _, at, col, mk = _t(kind)
    n = conn.execute(f"DELETE FROM {at} WHERE {col}=?", (pid,)).rowcount
    conn.execute("DELETE FROM meta WHERE key IN (?,?)", (f"{mk}:{pid}", user_key(kind, pid)))
    conn.commit()
    return n


def apply(conn, pos, X, ids_by_row, pid, thr, kind="place", user=False) -> int:
    """しきい値以上の写真を AI推定の表に入れ直す。入れた枚数を返す。しきい値は meta に保存する。
    user=True（人が押したとき）は「自動で付けた」記録も残す。"""
    _, at, col, mk = _t(kind)
    sc = scores(conn, pos, X, pid, kind)
    conn.execute(f"DELETE FROM {at} WHERE {col}=?", (pid,))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (f"{mk}:{pid}", str(thr)))
    if user:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (user_key(kind, pid), time.strftime("%Y-%m-%d")))
    n = 0
    if sc is not None:
        ok = _real_images(conn) - _excluded(conn, pid, kind)
        rows = [(pid, int(ids_by_row[i]), float(sc[i])) for i in range(len(sc)) if sc[i] >= thr and int(ids_by_row[i]) in ok]
        conn.executemany(f"INSERT OR REPLACE INTO {at}({col}, asset_id, sim) VALUES(?,?,?)", rows)
        n = len(rows)
    conn.commit()
    return n


def threshold(conn, pid, kind="place") -> float:
    r = conn.execute("SELECT value FROM meta WHERE key=?", (f"{_t(kind)[3]}:{pid}",)).fetchone()
    try:
        return float(r[0]) if r else DEFAULT_THR
    except ValueError:
        return DEFAULT_THR


def label(conn, asset_ids, pid, mode, kind="place") -> int:
    """mode: add（この場所・場面）/ neg（ではない）/ remove（付けたのを取り消す）"""
    lt, at, col, _ = _t(kind)
    stamp = time.strftime("%Y-%m-%d")
    n = 0
    for a in asset_ids:
        if mode == "remove":
            n += conn.execute(f"DELETE FROM {lt} WHERE asset_id=? AND {col}=?", (a, pid)).rowcount
        else:
            conn.execute(f"INSERT OR REPLACE INTO {lt}(asset_id,{col},positive,created) VALUES(?,?,?,?)", (a, pid, 1 if mode == "add" else 0, stamp))
            if mode == "neg":
                conn.execute(f"DELETE FROM {at} WHERE {col}=? AND asset_id=?", (pid, a))
            n += 1
    conn.commit()
    return n
