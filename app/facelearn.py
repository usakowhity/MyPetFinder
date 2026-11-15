"""顔の名前付け: 人が付けた少数の顔と似ている顔を探して、一括で付ける。仕組みは placelearn（場所）と同じ。

 ・類似度 = 人が付けた顔（その人の見本）との、コサイン類似度の上位3つの平均（SFace の特徴）
 ・グループ: まだ付いていない顔を、似たもの同士にまとめる（1つずつではなく、グループごとに名前を付けられる）
 ・「この人ではない」と答えた顔は、自動では付けない。1つの顔には、1人の名前だけを自動で付ける
"""
from __future__ import annotations

import time

K = 3
DEFAULT_THR = 0.45       # SFace のコサイン類似度。公式の目安（同一人物）は 0.363 以上。子どもは成長で変わるので、見本を増やして使う
CLUSTER_THR = 0.5


def load(conn):
    """(face_id の配列, asset_id の配列, 行列)。"""
    import numpy as np

    rows = conn.execute("SELECT id, asset_id, vec FROM faces WHERE vec IS NOT NULL ORDER BY id").fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    aids = np.array([r["asset_id"] for r in rows], dtype=np.int64)
    F = np.stack([np.frombuffer(r["vec"], dtype=np.float16) for r in rows]).astype(np.float32)
    return ids, aids, F


def _labels(conn, pid, positive):
    return [r[0] for r in conn.execute("SELECT face_id FROM face_labels WHERE person_id=? AND positive=?", (pid, positive))]


def _named(conn):
    """どこかの人物に、人が名前を付けた顔。"""
    return {r[0] for r in conn.execute("SELECT face_id FROM face_labels WHERE positive=1")}


def _ignored(conn):
    """検索に使わないと決めた顔（人形・他人など）。"""
    return {r[0] for r in conn.execute("SELECT face_id FROM face_ignore")}


def ignore(conn, face_ids, on=True) -> int:
    """face_ids を「使わない顔」にする（on=False で戻す）。AI推定からも外す。"""
    stamp = time.strftime("%Y-%m-%d")
    n = 0
    for f in face_ids:
        f = int(f)
        if on:
            n += conn.execute("INSERT OR IGNORE INTO face_ignore(face_id,created) VALUES(?,?)", (f, stamp)).rowcount
            conn.execute("DELETE FROM face_ai WHERE face_id=?", (f,))
        else:
            n += conn.execute("DELETE FROM face_ignore WHERE face_id=?", (f,)).rowcount
    conn.commit()
    return n


def scores(conn, ids, F, pid):
    import numpy as np

    pos = {int(f): i for i, f in enumerate(ids)}
    ex = [pos[f] for f in _labels(conn, pid, 1) if f in pos]
    if not ex:
        return None
    S = F @ F[ex].T
    k = min(K, S.shape[1])
    top = np.partition(S, S.shape[1] - k, axis=1)[:, -k:] if S.shape[1] > k else S
    out = top.mean(axis=1)
    out[ex] = 1.0
    return out


def suggest(conn, ids, F, pid, limit=60, offset=0):
    import numpy as np

    sc = scores(conn, ids, F, pid)
    if sc is None:
        return [], 0
    skip = _named(conn) | _ignored(conn) | set(_labels(conn, pid, 0))
    cand = [(int(ids[i]), float(sc[i])) for i in np.argsort(-sc) if int(ids[i]) not in skip]
    return cand[offset: offset + limit], len(cand)


def review(conn, ids, F, pid, kind="human", limit=60, offset=0):
    """付けた名前の見直し用。kind=human: 人が付けた顔を「他の見本と似ていない順」（間違いの疑いが強い順）、
    kind=ai: 自動で付いた顔を「似ていない順」。[(face_id, sim)], 総数"""
    import numpy as np

    pos = {int(f): i for i, f in enumerate(ids)}
    if kind == "ai":
        rows = conn.execute("SELECT face_id, sim FROM face_ai WHERE person_id=? ORDER BY sim ASC, face_id", (pid,)).fetchall()
        rows = [(int(r[0]), float(r[1] if r[1] is not None else 0)) for r in rows if int(r[0]) not in _ignored(conn)]
        return rows[offset: offset + limit], len(rows)
    lab = [f for f in _labels(conn, pid, 1) if f in pos]
    if not lab:
        return [], 0
    idx = [pos[f] for f in lab]
    if len(lab) < 2:
        sc = np.ones(len(lab))
    else:
        S = F[idx] @ F[idx].T
        np.fill_diagonal(S, -1)
        k = min(K, len(lab) - 1)
        sc = np.sort(S, axis=1)[:, -k:].mean(axis=1)
    order = np.argsort(sc)
    rows = [(int(lab[i]), float(sc[i])) for i in order]
    return rows[offset: offset + limit], len(rows)


def count_at(conn, ids, F, pid, thr):
    sc = scores(conn, ids, F, pid)
    if sc is None:
        return 0
    skip = _named(conn) | _ignored(conn) | set(_labels(conn, pid, 0))
    return sum(1 for i in range(len(sc)) if sc[i] >= thr and int(ids[i]) not in skip)


def loo_recall(conn, ids, F, pid, thr):
    import numpy as np

    pos = {int(f): i for i, f in enumerate(ids)}
    ex = [pos[f] for f in _labels(conn, pid, 1) if f in pos]
    if len(ex) < 3:
        return None
    S = F[ex] @ F[ex].T
    np.fill_diagonal(S, -1)
    k = min(K, len(ex) - 1)
    return float((np.sort(S, axis=1)[:, -k:].mean(axis=1) >= thr).mean())


def apply_all(conn, ids, F, thr_of) -> dict:
    """見本のある全員について、自動で付け直す。1つの顔には、最も似ている1人だけ。thr_of(person_id) → しきい値。{person_id: 件数}"""
    import numpy as np

    people = [r[0] for r in conn.execute("SELECT DISTINCT person_id FROM face_labels WHERE positive=1")]
    conn.execute("DELETE FROM face_ai")
    if not people or len(ids) == 0:
        conn.commit()
        return {}
    named = _named(conn) | _ignored(conn)
    best = np.full(len(ids), -1.0)
    who = np.full(len(ids), -1, dtype=np.int64)
    thr = np.array([thr_of(p) for p in people])
    for j, p in enumerate(people):
        sc = scores(conn, ids, F, p)
        if sc is None:
            continue
        neg = {f for f in _labels(conn, p, 0)}
        ok = (sc >= thr[j]) & (sc > best)
        for i in np.nonzero(ok)[0]:
            if int(ids[i]) not in neg:
                best[i], who[i] = sc[i], j
    rows, cnt = [], {}
    for i in np.nonzero(who >= 0)[0]:
        if int(ids[i]) in named:
            continue
        p = people[who[i]]
        rows.append((p, int(ids[i]), float(best[i])))
        cnt[p] = cnt.get(p, 0) + 1
    conn.executemany("INSERT OR REPLACE INTO face_ai(person_id, face_id, sim) VALUES(?,?,?)", rows)
    conn.commit()
    return cnt


def threshold(conn, pid) -> float:
    r = conn.execute("SELECT value FROM meta WHERE key=?", (f"face_thr:{pid}",)).fetchone()
    try:
        return float(r[0]) if r else DEFAULT_THR
    except ValueError:
        return DEFAULT_THR


def set_threshold(conn, pid, thr):
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (f"face_thr:{pid}", str(thr)))
    conn.commit()


def label(conn, face_ids, pid, mode) -> int:
    """mode: add / neg / remove"""
    stamp = time.strftime("%Y-%m-%d")
    n = 0
    for f in face_ids:
        if mode == "remove":
            n += conn.execute("DELETE FROM face_labels WHERE face_id=? AND person_id=?", (f, pid)).rowcount
        else:
            if mode == "add":                                              # 1つの顔は、1人だけ（付け直しでは、前の人の名前を外す）
                conn.execute("DELETE FROM face_labels WHERE face_id=? AND positive=1 AND person_id<>?", (f, pid))
            conn.execute("INSERT OR REPLACE INTO face_labels(face_id,person_id,positive,created) VALUES(?,?,?,?)", (f, pid, 1 if mode == "add" else 0, stamp))
            conn.execute("DELETE FROM face_ai WHERE face_id=?", (f,))          # 人が決めた顔は、AI推定から外す
            n += 1
    conn.commit()
    return n


def clusters(conn, ids, F, thr=CLUSTER_THR, min_size=2, top=40, sample=16):
    """名前の付いていない顔を、似たもの同士にまとめる。大きいグループ順の [{size, ids, sample}]。"""
    import numpy as np

    named = _named(conn) | _ignored(conn) | {r[0] for r in conn.execute("SELECT face_id FROM face_labels WHERE positive=0")}
    idx = [i for i in range(len(ids)) if int(ids[i]) not in named]
    if not idx:
        return []
    L = np.zeros((min(len(idx), 20000), F.shape[1]), dtype=np.float32)
    members, nl = [], 0
    for i in idx:
        v = F[i]
        if nl:
            s = L[:nl] @ v
            j = int(np.argmax(s))
            if s[j] >= thr:
                members[j].append(i)
                continue
        if nl >= len(L):
            continue
        L[nl] = v
        members.append([i])
        nl += 1
    out = []
    for j, m in enumerate(members):
        if len(m) < min_size:
            continue
        s = F[m] @ L[j]
        order = np.argsort(-s)
        out.append({"size": len(m), "ids": [int(ids[m[k]]) for k in order], "sample": [int(ids[m[k]]) for k in order[:sample]]})
    out.sort(key=lambda c: -c["size"])
    return out[:top]
