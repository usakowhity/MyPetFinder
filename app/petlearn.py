"""ペットの「グループで名前を付ける」: 人物（顔）と同じ操作で、ペットに名前を付ける。

 ・特徴: DINOv2（個体の見分けに強い）。無ければ CLIP。
 ・名前付け = file_pets に origin='user', status='identified'（その画像のファイルすべて）。これが、判定の正解データになる。
 ・外す・違う = status='rejected', origin='user'（フォルダ名から付いた名前が間違っていたときも、これで打ち消せる）。
 ・「使わない」= pet_ignore（人形・写真集など、ペットとして数えない画像）。グループ・候補に出さない。
 ・名前付けのあと、広げるのは「判定をやり直す」（メニュー11と同じ kNN + 撮影日・生存期間）。
"""
from __future__ import annotations

import time
from collections import defaultdict

from . import embed
from .petcheck import collect_labels

K = 3


def model_of(conn) -> str:
    return embed.pet_model(conn)


def load(conn):
    """(asset_id の配列, 行列)。実写の画像だけ。"""
    import numpy as np

    ids, X = embed.load_matrix(conn, model=model_of(conn))
    if len(ids) == 0:
        return ids, X
    real = {r[0] for r in conn.execute("SELECT id FROM assets WHERE zone='real' AND kind='image'")}
    keep = np.array([int(a) in real for a in ids])
    return ids[keep], X[keep]


def _ignored(conn):
    return {r[0] for r in conn.execute("SELECT asset_id FROM pet_ignore")}


def ignore(conn, asset_ids, on=True) -> int:
    n = 0
    for a in asset_ids:
        a = int(a)
        if on:
            n += conn.execute("INSERT OR IGNORE INTO pet_ignore(asset_id,created) VALUES(?,?)", (a, time.strftime("%Y-%m-%d"))).rowcount
        else:
            n += conn.execute("DELETE FROM pet_ignore WHERE asset_id=?", (a,)).rowcount
    conn.commit()
    return n


def _files(conn, aid):
    return [r[0] for r in conn.execute("SELECT id FROM files WHERE asset_id=? AND status='present'", (int(aid),))]


def label(conn, asset_ids, pid, mode="add") -> int:
    """mode: add = この子です / neg = この子ではない / remove = 名前を外す（どちらも rejected にする）。"""
    n = 0
    for a in asset_ids:
        for fid in _files(conn, a):
            if mode == "add":
                conn.execute("""INSERT INTO file_pets(file_id,pet_id,status,strength,origin,reason) VALUES(?,?,'identified','strong','user','pet ui')
                                ON CONFLICT(file_id,pet_id) DO UPDATE SET status='identified', strength='strong', origin='user', reason='pet ui'""", (fid, pid))
            else:
                conn.execute("""INSERT INTO file_pets(file_id,pet_id,status,strength,origin,reason) VALUES(?,?,'rejected',NULL,'user','pet ui')
                                ON CONFLICT(file_id,pet_id) DO UPDATE SET status='rejected', strength=NULL, origin='user', reason='pet ui'""", (fid, pid))
        n += 1
    if mode == "add":
        ph = ",".join("?" * len(asset_ids)) or "NULL"
        conn.execute(f"DELETE FROM pet_ignore WHERE asset_id IN ({ph})", [int(a) for a in asset_ids])
    conn.commit()
    return n


def move(conn, asset_ids, old_pid, new_pid) -> int:
    """付け直し: 前のペットの名前を外し（rejected）、新しいペットを付ける。"""
    label(conn, asset_ids, old_pid, "remove")
    return label(conn, asset_ids, new_pid, "add")


def labelled(conn):
    """{asset_id: pet_id}。単独のペットが付いた画像（名前・人の確認）。"""
    return {a: v[0] for a, v in collect_labels(conn).items()}


def rejected(conn, pid):
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT f.asset_id FROM file_pets fp JOIN files f ON f.id=fp.file_id WHERE fp.pet_id=? AND fp.status='rejected' AND f.asset_id IS NOT NULL", (pid,))}


def topk_mean(Q, E, k=K, self_rows=None, chunk=1500):
    """各行 Q[i] について、E との類似度の上位 k 件の平均。self_rows[i] = E の中の自分自身の行（除く）。なければ None。"""
    import numpy as np

    out = np.zeros(len(Q), dtype=np.float32)
    if len(E) == 0:
        return out
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ E.T
        if self_rows is not None:
            r = np.arange(S.shape[0])
            rows = self_rows[s:s + chunk]
            m = rows >= 0
            S[r[m], rows[m]] = -1.0
        kk = min(k, S.shape[1])
        top = np.partition(S, S.shape[1] - kk, axis=1)[:, -kk:] if S.shape[1] > kk else S
        out[s:s + chunk] = top.mean(axis=1)
    return out


def leader_groups(X, thr, seed=0, block=512):
    """似たものを同じグループにまとめる（先頭と thr 以上似ていれば同じグループ）。各行のグループ番号を返す。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    assign = np.full(len(X), -1, dtype=int)
    L = np.zeros((0, X.shape[1]), dtype=X.dtype)
    for s in range(0, len(order), block):
        idx = order[s:s + block]
        if len(L):
            S = X[idx] @ L.T
            j = S.argmax(axis=1)
            ok = S[np.arange(len(idx)), j] >= thr
            assign[idx[ok]] = j[ok]
            rest = idx[~ok]
        else:
            rest = idx
        for i in rest:                                                  # 残りは、順に新しいグループを作る
            if len(L):
                sc = L @ X[i]
                j = int(sc.argmax())
                if sc[j] >= thr:
                    assign[i] = j
                    continue
            assign[i] = len(L)
            L = np.vstack([L, X[i][None, :]])
    return assign


def groups(conn, ids, X, thr=None, min_size=5, sample=12, limit=400):
    """名前の付いていない画像のグループ。[{size, ids(全部), sample(代表)}]、大きい順。"""
    import numpy as np

    thr = thr if thr is not None else (0.6 if model_of(conn) == embed.DINO_MODEL else 0.85)
    named = set(labelled(conn)) | _ignored(conn)
    rows = np.array([i for i, a in enumerate(ids) if int(a) not in named], dtype=int)
    if len(rows) == 0:
        return []
    Xs = X[rows]
    a = leader_groups(Xs, thr)
    by = defaultdict(list)
    for i, g in enumerate(a):
        by[g].append(i)
    out = []
    for g, members in by.items():
        if len(members) < min_size:
            continue
        c = Xs[members].mean(axis=0)
        sims = Xs[members] @ (c / (np.linalg.norm(c) + 1e-12))
        order = np.argsort(-sims)
        asset = [int(ids[rows[members[o]]]) for o in order]
        out.append({"size": len(asset), "ids": asset[:3000], "sample": asset[:sample]})
    out.sort(key=lambda c: -c["size"])
    return out[:limit]


def examples(conn, ids, pid):
    pos = {int(a): i for i, a in enumerate(ids)}
    return [a for a, p in labelled(conn).items() if p == pid and a in pos], pos


def suggest(conn, ids, X, pid, limit=150, offset=0):
    """pid に似た、名前の付いていない画像（似ている順）。[(asset_id, sim)], 件数"""
    import numpy as np

    ex, pos = examples(conn, ids, pid)
    if not ex:
        return [], 0
    skip = set(labelled(conn)) | _ignored(conn) | rejected(conn, pid)
    rows = [i for i, a in enumerate(ids) if int(a) not in skip]
    if not rows:
        return [], 0
    sc = topk_mean(X[rows], X[[pos[a] for a in ex]])
    order = np.argsort(-sc)
    items = [(int(ids[rows[o]]), float(sc[o])) for o in order]
    return items[offset: offset + limit], len(items)


def review(conn, ids, X, pid, kind="human", limit=120, offset=0):
    """付けた名前の見直し。kind=human: 名前・人の確認で付いた画像を「他の見本と似ていない順」に。
    kind=ai: 自動の判定（file_pets origin='ai'）の画像を、見本との類似度の低い順に。[(asset_id, sim)], 件数"""
    import numpy as np

    ex, pos = examples(conn, ids, pid)
    if not ex:
        return [], 0
    E = X[[pos[a] for a in ex]]
    if kind == "ai":
        cand = [int(r[0]) for r in conn.execute(
            """SELECT DISTINCT f.asset_id FROM file_pets fp JOIN files f ON f.id=fp.file_id
               WHERE fp.pet_id=? AND fp.origin='ai' AND f.status='present' AND f.asset_id IS NOT NULL""", (pid,))]
        cand = [a for a in cand if a in pos and a not in set(ex) and a not in _ignored(conn)]
        sc = topk_mean(X[[pos[a] for a in cand]], E) if cand else np.zeros(0)
        items = sorted(zip(cand, map(float, sc)), key=lambda t: t[1])
    else:
        selfrows = np.arange(len(ex))
        sc = topk_mean(E, E, self_rows=selfrows if len(ex) > 1 else None)
        items = sorted(zip(ex, map(float, sc)), key=lambda t: t[1])
    return items[offset: offset + limit], len(items)


def pet_list(conn):
    """[{id, name, species, labelled, auto}]。"""
    lab = defaultdict(int)
    for p in labelled(conn).values():
        lab[p] += 1
    auto = {r[0]: r[1] for r in conn.execute(
        """SELECT fp.pet_id, COUNT(DISTINCT f.asset_id) FROM file_pets fp JOIN files f ON f.id=fp.file_id
           WHERE fp.origin='ai' AND f.status='present' AND f.asset_id IS NOT NULL GROUP BY fp.pet_id""")}
    return [{"id": r["id"], "name": r["display_name"] or r["id"], "species": r["species"] or "", "labelled": lab.get(r["id"], 0), "auto": auto.get(r["id"], 0)}
            for r in conn.execute("SELECT id, display_name, species FROM pets WHERE active=1 ORDER BY display_name")]
