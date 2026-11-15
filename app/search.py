"""自然言語（日本語）の検索文を、条件に分けて検索する。クラウドのAIは使わない。

条件の状態:
  used    … 判定済みのデータで、実際に絞り込んでいる（ペット名・場所・タグ・日付・写真/動画・遺影など）
  approx  … 近似（場面の言葉を、CLIP の画像と文章の近さで並べ替える）
  pending … まだ判定できない（人物など）。検索には使わず、画面に灰色で表示する
  unknown … 解釈できなかった語
"""
from __future__ import annotations

import re
from pathlib import Path

from .textnorm import norm

PERSON_WORDS = ["孫", "子供", "子ども", "こども", "赤ちゃん", "家族", "友達", "友だち", "夫婦"]
VIDEO_WORDS = ["ビデオ", "動画", "ムービー", "映像"]
IMAGE_WORDS = ["写真", "画像", "スナップ"]
NONLIVE_WORDS = {"遺影": "framed", "額入り": "framed", "置物": "figurine", "縫いぐるみ": "figurine", "ぬいぐるみ": "figurine", "画面": "screen"}
NONLIVE_LABEL = {"framed": "遺影・額入り写真", "figurine": "置物・縫いぐるみ", "screen": "画面"}
SEASONS = {"春": [3, 4, 5], "夏": [6, 7, 8], "秋": [9, 10, 11], "冬": [12, 1, 2]}
# 場面の言葉 → 英語（CLIP は英語の文章で画像と比べるため）。順に探し、当たったものを文章にまとめる
SCENE_WORDS = [
    ("仲良", "close together, friendly", "仲良く"), ("遊", "playing", "遊ぶ"), ("食べ", "eating", "食べる"), ("食事", "eating", "食事"), ("寝", "sleeping", "寝る"),
    ("眠", "sleeping", "眠る"), ("走", "running", "走る"), ("跳", "jumping", "跳ぶ"), ("抱", "being held in arms", "抱っこ"),
    ("撫で", "being petted", "撫でる"), ("なで", "being petted", "撫でる"), ("散歩", "walking outdoors", "散歩"),
    ("日向ぼっこ", "lying in the sun", "日向ぼっこ"), ("ケージ", "in a cage", "ケージ"), ("庭", "in a garden", "庭"), ("雪", "in snow", "雪"),
    ("花", "with flowers", "花"), ("海", "at the sea", "海"), ("おもちゃ", "with a toy", "おもちゃ"), ("ご飯", "eating", "ご飯"),
    ("くつろ", "relaxing", "くつろぐ"), ("甘え", "cuddling", "甘える"), ("じゃれ", "playfully", "じゃれる"), ("見つめ", "looking at the camera", "見つめる"),
]
STOP = ["様子", "写した", "撮った", "写って", "写る", "している", "していた", "ている", "いる", "して", "たち", "達", "みんな", "一緒", "の", "を", "が", "は", "に", "で", "と", "も", "や", "から", "まで", "ます", "ました", "です"]
_KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _take(s: str, needle: str):
    n = norm(needle)
    if n and n in s:
        return s.replace(n, " ", 1), True
    return s, False


def _vocab(conn, main, alias, idcol, aliascol, namecol):
    names = {}
    ids = {}
    for r in conn.execute(f"SELECT * FROM {main}"):
        ids[r["id"]] = r[namecol] or r["id"]
        names[norm(r["id"])] = r["id"]
        if r[namecol]:
            names[norm(r[namecol])] = r["id"]
    for r in conn.execute(f"SELECT * FROM {alias}"):
        names[r["alias_norm"]] = r[aliascol]
    items = [(k, v) for k, v in names.items() if len(k) >= 2 and v in ids]
    items.sort(key=lambda kv: -len(kv[0]))
    return items, ids


def load_scene_words(cfg) -> list:
    """config/scene_words.toml があれば、場面の言葉の追加・上書き（words = [["遊", "playing"], ...]）を使う。"""
    try:
        from .config import load_toml

        p = Path(cfg.config_dir) / "scene_words.toml"
        if p.exists():
            extra = [(w[0], w[1], w[2] if len(w) > 2 else w[0]) for w in load_toml(p).get("words", [])]
            return extra + SCENE_WORDS
    except Exception:
        pass
    return SCENE_WORDS


def parse(conn, q: str, scene_words=None) -> dict:
    scene_words = scene_words or SCENE_WORDS
    s = " " + norm(q) + " "
    conds = []

    def add(kind, label, status, detail="", **kw):
        conds.append({"kind": kind, "label": label, "status": status, "detail": detail, **kw})

    # 人数（「3人の孫たち」）
    count = None
    m = re.search(r"(\d+|[一二三四五六七八九十]+)\s*人", s)
    if m:
        g = m.group(1)
        count = int(g) if g.isdigit() else _KANJI_NUM.get(g[0], None)
        s = s[:m.start()] + " " + s[m.end():]
    # 写真・動画
    media = None
    for w in VIDEO_WORDS:
        s, ok = _take(s, w)
        media = "video" if ok else media
    for w in IMAGE_WORDS:
        s, ok = _take(s, w)
        if ok and media is None:
            media = "image"
    if media == "video":
        add("media", "動画", "used", "動画だけに絞り込み（動画は、コマの判定をまとめて使います。メニュー17で取り込んだものが対象です）", value="video")
    elif media == "image":
        add("media", "写真", "used", "写真だけに絞り込み", value="image")
    # 遺影・置物・画面
    nonlive = []
    for w, k in NONLIVE_WORDS.items():
        s2, ok = _take(s, w)
        if ok:
            s = s2
            if k not in nonlive:
                nonlive.append(k)
    for k in nonlive:
        add("nonlive", NONLIVE_LABEL[k], "used", "生きたペットではない写真の候補（撮影日・登録フォルダ・ゼロショット判定）で絞り込み", value=k)
    # 日付
    dates = []
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s):
        dates.append(("ym", int(m.group(1)), int(m.group(2))))
    s = re.sub(r"(\d{4})\s*年\s*(\d{1,2})\s*月", " ", s)
    for m in re.finditer(r"(\d{4})\s*年", s):
        dates.append(("y", int(m.group(1)), None))
    s = re.sub(r"(\d{4})\s*年", " ", s)
    for m in re.finditer(r"(\d{1,2})\s*月", s):
        dates.append(("m", None, int(m.group(1))))
    s = re.sub(r"(\d{1,2})\s*月", " ", s)
    for name, months in SEASONS.items():
        s2, ok = _take(s, name)
        if ok:
            s = s2
            dates.append(("season", name, months))
    for d in dates:
        if d[0] == "ym":
            add("date", f"{d[1]}年{d[2]}月", "used", "撮影日で絞り込み（撮影日が不明の写真は含みません）", ym=[d[1], d[2]])
        elif d[0] == "y":
            add("date", f"{d[1]}年", "used", "撮影日で絞り込み（撮影日が不明の写真は含みません）", year=d[1])
        elif d[0] == "m":
            add("date", f"{d[2]}月", "used", "撮影日で絞り込み（何年でも）", months=[d[2]])
        else:
            add("date", d[1], "used", f"撮影日で絞り込み（{'・'.join(str(x) for x in d[2])}月）", months=d[2])
    # ペット
    items, ids = _vocab(conn, "pets", "pet_aliases", "id", "pet_id", "display_name")
    pets = []
    for k, pid in items:
        s2, ok = _take(s, k)
        if ok:
            s = s2
            if pid not in pets:
                pets.append(pid)
    for pid in pets:
        add("pet", ids[pid], "used", "ペットの判定（名前・AI・確認済み）で絞り込み", id=pid)
    # 人物（台帳の名前・呼称。「孫」のように、1つの呼称を複数の人に付けておくと、その誰かが写っている写真を探す）
    amap, dname = {}, {}
    for r in conn.execute("SELECT id, display_name FROM persons WHERE active=1"):
        dname[r["id"]] = r["display_name"] or r["id"]
        for k in (norm(r["id"]), norm(r["display_name"])):
            if k:
                amap.setdefault(k, set()).add(r["id"])
    for r in conn.execute("SELECT person_id, alias_norm FROM person_aliases"):
        if r["person_id"] in dname and r["alias_norm"]:
            amap.setdefault(r["alias_norm"], set()).add(r["person_id"])
    cjk1 = lambda k: len(k) == 1 and "\u4e00" <= k <= "\u9fff"
    seen_sets = []
    handled = set()
    labelled = {r[0] for r in conn.execute("SELECT DISTINCT person_id FROM face_labels WHERE positive=1")} if has_table(conn, "face_labels") else set()
    for k in sorted((k for k in amap if len(k) >= 2 or cjk1(k)), key=lambda k: -len(k)):
        s2, ok = _take(s, k)
        if not ok:
            continue
        s = s2
        ids_ = sorted(amap[k])
        handled.add(k)
        if ids_ in seen_sets:
            continue
        seen_sets.append(ids_)
        label = dname[ids_[0]] if len(ids_) == 1 else k
        n_ = count if (count and len(seen_sets) == 1) else None
        used = any(i in labelled for i in ids_)
        add("person", label + (f" ×{n_}" if n_ else ""), "used" if used else "pending",
            "顔の名前付けで絞り込み（人が付けた顔と、似ている顔のAI推定）" if used else "顔の見本がまだありません。検索画面の「顔に名前を付ける」で、この人の顔を何枚か付けると使えます",
            ids=ids_, count=n_)
    words = []
    for w in PERSON_WORDS:
        if norm(w) in handled:
            continue
        s2, ok = _take(s, w)
        if ok:
            s = s2
            words.append(w)
    for w in words:
        add("person", w + (f" ×{count}" if count else ""), "pending", "人物の登録がありません（persons.toml の呼称に追加し、顔に名前を付けると使えます）")
    if count and not words and not seen_sets:
        add("person", f"{count}人", "pending", "人数の判定には、人物の登録と顔の名前付けが必要です")
    # 場所・タグ
    for main, alias, kind, namecol in (("places", "place_aliases", "place", "name"), ("tags", "tag_aliases", "tag", "name")):
        items, ids2 = _vocab(conn, main, alias, "id", f"{kind}_id", namecol)
        got = []
        for k, tid in items:
            s2, ok = _take(s, k)
            if ok:
                s = s2
                if tid not in got:
                    got.append(tid)
        for tid in got:
            add(kind, ids2[tid], "used", ("場所の台帳・フォルダ名から付いた場所" if kind == "place" else "フォルダ名・ファイル名から付いたタグ") + "で絞り込み", id=tid)
    # 場面（近似）
    parts, shown = [], []
    for w, e, disp in scene_words:
        s2, ok = _take(s, w)
        if ok:
            s = s2
            if e not in parts:
                parts.append(e)
                shown.append(disp)
    if parts:
        add("scene", "・".join(shown), "approx", "場面: CLIP の画像と文章の近さで並べ替え（近似）。写真だけが対象です", parts=parts)
    # 残った語
    rest = s
    for w in sorted(STOP, key=len, reverse=True):
        rest = rest.replace(norm(w), " ")
    for t in re.split(r"[\s、。,.!?！？「」『』（）()・]+", rest):
        if len(t) >= 2 or (t and not re.fullmatch(r"[ぁ-んー]", t)):
            if t:
                add("unknown", t, "unknown", "解釈できなかった語（検索には使いません）")
    return {"query": q, "conds": conds, "species": [r["species"] for r in conn.execute("SELECT species FROM pets WHERE id IN (%s)" % ",".join("?" * len(pets)), pets)] if pets else []}


# ---------------------------------------------------------------- 検索
CONFIDENT = "(p.status='identified' OR (p.status='hint' AND p.strength IN ('strong','high','mid')))"


def _where(parsed: dict, opts: dict):
    where, args = ["1=1"], []
    if not opts.get("include_ai"):
        where.append("a.zone='real'")
    pets = [c["id"] for c in parsed["conds"] if c["kind"] == "pet"]
    for pid in pets:
        cond = "p.pet_id=? AND p.status<>'rejected'"
        if not opts.get("loose"):
            cond += " AND " + CONFIDENT
        where.append(f"a.id IN (SELECT f2.asset_id FROM file_pets p JOIN files f2 ON f2.id=p.file_id WHERE {cond} AND f2.asset_id IS NOT NULL)")
        args.append(pid)
    for c in parsed["conds"]:
        k = c["kind"]
        if k == "place":
            ai = "" if opts.get("no_ai_place") else " UNION SELECT asset_id FROM place_ai WHERE place_id=?"
            where.append("(a.id IN (SELECT f2.asset_id FROM file_places x JOIN files f2 ON f2.id=x.file_id WHERE x.place_id=? AND f2.asset_id IS NOT NULL"
                         " UNION SELECT asset_id FROM place_labels WHERE place_id=? AND positive=1" + ai + ")"
                         " AND a.id NOT IN (SELECT asset_id FROM place_labels WHERE place_id=? AND positive=0))")
            args += [c["id"], c["id"]] + ([] if opts.get("no_ai_place") else [c["id"]]) + [c["id"]]
        elif k == "tag":
            ai = "" if opts.get("no_ai_place") else " UNION SELECT asset_id FROM tag_ai WHERE tag_id=?"
            where.append("(a.id IN (SELECT f2.asset_id FROM file_tags x JOIN files f2 ON f2.id=x.file_id WHERE x.tag_id=? AND f2.asset_id IS NOT NULL"
                         " UNION SELECT asset_id FROM tag_labels WHERE tag_id=? AND positive=1" + ai + ")"
                         " AND a.id NOT IN (SELECT asset_id FROM tag_labels WHERE tag_id=? AND positive=0))")
            args += [c["id"], c["id"]] + ([] if opts.get("no_ai_place") else [c["id"]]) + [c["id"]]
        elif k == "person" and c["status"] == "used":
            ids_ = c["ids"]
            q = ",".join("?" * len(ids_))
            src = (f"SELECT f.asset_id aid, l.person_id pid FROM face_labels l JOIN faces f ON f.id=l.face_id WHERE l.positive=1 AND l.person_id IN ({q})"
                   f" UNION SELECT f.asset_id, i.person_id FROM face_ai i JOIN faces f ON f.id=i.face_id WHERE i.person_id IN ({q})")
            if c.get("count"):
                where.append(f"a.id IN (SELECT aid FROM ({src}) GROUP BY aid HAVING COUNT(DISTINCT pid) >= ?)")
                args += ids_ + ids_ + [c["count"]]
            else:
                where.append(f"a.id IN (SELECT aid FROM ({src}))")
                args += ids_ + ids_
        elif k == "media":
            where.append("a.kind=?")
            args.append(c["value"])
        elif k == "scene":
            pass
    yr, mo = [], []
    for c in parsed["conds"]:
        if c["kind"] != "date":
            continue
        if "ym" in c:
            yr.append(("substr(a.date_taken,1,7)=?", "%04d-%02d" % tuple(c["ym"])))
        elif "year" in c:
            yr.append(("substr(a.date_taken,1,4)=?", "%04d" % c["year"]))
        else:
            mo += ["%02d" % m for m in c["months"]]
    if yr:
        where.append("(" + " OR ".join(w for w, _ in yr) + ")")
        args += [a for _, a in yr]
    if mo:
        mo = sorted(set(mo))
        where.append("substr(a.date_taken,6,2) IN (%s)" % ",".join("?" * len(mo)))
        args += mo
    nl = [c["value"] for c in parsed["conds"] if c["kind"] == "nonlive"]
    if nl:
        # 遺影・置物・画面: 撮影日・登録フォルダの印（asset_nonlive）か、ゼロショット判定の種類
        zs = {"framed": ("framed",), "figurine": ("figurine",), "screen": ("screen",)}
        tops = sorted({t for k in nl for t in zs[k]})
        if opts.get("zs", True):
            where.append("(a.id IN (SELECT asset_id FROM asset_nonlive)"
                         " OR a.id IN (SELECT asset_id FROM asset_zeroshot WHERE top IN (%s)))" % ",".join("?" * len(tops)))
            args += tops
        else:
            where.append("a.id IN (SELECT asset_id FROM asset_nonlive)")
    elif pets and not opts.get("show_nonlive"):
        # ペット名で探すときは、既定で「生きた姿ではない」候補（遺影・置物など）を除く
        where.append("a.id NOT IN (SELECT asset_id FROM asset_nonlive)")
    return " AND ".join(where), args


def has_table(conn, name) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


class Searcher:
    """検索の実行。画像の特徴行列と文章モデルは、必要になった時に1回だけ読み込む。"""

    def __init__(self, cfg, conn):
        self.cfg, self.conn = cfg, conn
        self._mat = None
        self._vf = None
        self._text = {}
        self.scene_words = load_scene_words(cfg)

    def parse(self, q):
        return parse(self.conn, q, self.scene_words)

    def warm(self):
        """検索画面を開いたあとの最初の場面検索を速くするため、裏で特徴行列と文章モデルを読み込んでおく（別の接続を使う）。"""
        try:
            from . import db, embed, zeroshot

            c = db.connect(self.cfg)
            try:
                ids, X = embed.load_matrix(c, with_video=True)
            finally:
                c.close()
            if self._mat is None:
                self._mat = ({int(a): i for i, a in enumerate(ids)}, X)
            md = self.cfg.app_root / "models" / embed.MODEL_NAME
            if (md / "text_model.onnx").exists():
                zeroshot._load(md)
        except Exception:
            pass

    def matrix(self):
        """(行の順の asset_id 配列, 行列)。場所の学習用。"""
        import numpy as np

        pos, X = self._matrix()
        ids = np.zeros(len(pos), dtype=np.int64)
        for a, i in pos.items():
            ids[i] = a
        return pos, X, ids

    def _matrix(self):
        if self._mat is None:
            from . import embed

            ids, X = embed.load_matrix(self.conn, with_video=True)
            self._mat = ({int(a): i for i, a in enumerate(ids)}, X)
        return self._mat

    def _frames(self):
        """動画のコマの特徴。({asset_id: (先頭行, 終わりの行)}, 行列)。"""
        if self._vf is None:
            from . import video

            ids, X = video.load_frames(self.conn)
            rng = {}
            for i, a in enumerate(ids):
                a = int(a)
                rng[a] = (rng[a][0], i + 1) if a in rng else (i, i + 1)
            self._vf = (rng, X)
        return self._vf

    def _text_vec(self, prompt):
        if prompt not in self._text:
            from . import embed, zeroshot

            self._text[prompt] = zeroshot.text_embeddings(self.cfg.app_root / "models" / embed.MODEL_NAME, [prompt])[0]
        return self._text[prompt]

    def search(self, q: str, opts: dict | None = None, limit=60, offset=0, ids_only=False) -> dict:
        import numpy as np

        opts = opts or {}
        parsed = self.parse(q)
        opts = {**opts, "zs": has_table(self.conn, "asset_zeroshot")}
        where, args = _where(parsed, opts)
        conds = parsed["conds"]
        usable = [c for c in conds if c["status"] in ("used", "approx")]
        if not usable:
            return {"parsed": parsed, "total": 0, "items": [], "note": "検索に使える条件がありません。ペット名・年・場所・「動画」などを入れてください。"}
        if not has_table(self.conn, "asset_nonlive"):
            return {"parsed": parsed, "total": 0, "items": [], "note": "先にメニュー14（NoPet）を実行してください。"}
        base = f"FROM assets a WHERE {where}"
        note = ""
        scene = next((c for c in conds if c["kind"] == "scene"), None)
        order = "ORDER BY a.date_taken IS NULL, a.date_taken " + ("ASC" if opts.get("oldest") else "DESC") + ", a.id"
        sim = {}
        if scene:
            ids = [r[0] for r in self.conn.execute(f"SELECT a.id {base}", args)]
            total_c = len(ids)
            try:
                pos, X = self._matrix()
                subj = "a pet rabbit" if "rabbit" in parsed["species"] or not parsed["species"] else "a pet rat"
                prompt = f"a photo of {subj}, " + ", ".join(scene["parts"])
                t = self._text_vec(prompt)
                have = [a for a in ids if a in pos]
                sc = X[[pos[a] for a in have]] @ t if have else np.zeros(0)
                rng, XF = self._frames()
                if rng:                                                   # 動画は、コマのうち文章に最も近いものの点数にする
                    sc = sc.copy()
                    for i, a in enumerate(have):
                        if a in rng:
                            sc[i] = float((XF[rng[a][0]: rng[a][1]] @ t).max())
                top = np.argsort(-sc)[: int(opts.get("scene_top", 300))]
                sim = {have[i]: float(sc[i]) for i in top}
                ordered = [have[i] for i in top]
                total = len(ordered)
                page = ordered[offset: offset + limit]
                note = f"場面は近似です（条件に合う {total_c:,} 枚のうち、文章に近い上位 {total} 枚）。"
            except Exception as ex:
                scene["status"] = "pending"
                scene["detail"] = f"文章モデルの準備ができないため使いません（{type(ex).__name__}）。メニュー15を一度実行してください。"
                note = scene["detail"]
                usable2 = [c for c in conds if c["status"] == "used"]
                if not usable2:
                    return {"parsed": parsed, "total": 0, "items": [], "note": note}
                total = self.conn.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
                page = [r[0] for r in self.conn.execute(f"SELECT a.id {base} {order} LIMIT ? OFFSET ?", args + [limit, offset])]
        else:
            total = self.conn.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
            page = [r[0] for r in self.conn.execute(f"SELECT a.id {base} {order} LIMIT ? OFFSET ?", args + [limit, offset])]
        if ids_only:
            return {"parsed": parsed, "total": total, "ids": list(page), "note": note}
        return {"parsed": parsed, "total": total, "items": self._items(page, sim), "note": note}

    def _items(self, ids, sim):
        if not ids:
            return []
        from .report import _mounts, abs_path

        mounts = _mounts(self.conn)
        qs = ",".join("?" * len(ids))
        info = {r["id"]: dict(r) for r in self.conn.execute(
            f"SELECT id, kind, date_taken, date_precision, duration_sec, width, height, n_files FROM assets WHERE id IN ({qs})", ids)}
        paths = {}
        for r in self.conn.execute(f"SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IN ({qs}) ORDER BY id DESC", ids):
            paths[r["asset_id"]] = abs_path(mounts, r["vol_id"], r["volpath"])
        pets = {}
        for r in self.conn.execute(
                f"""SELECT DISTINCT f.asset_id, p.pet_id FROM files f JOIN file_pets p ON p.file_id=f.id
                    WHERE f.asset_id IN ({qs}) AND p.status<>'rejected' AND {CONFIDENT}""", ids):
            pets.setdefault(r["asset_id"], []).append(r["pet_id"])
        places = {}
        for r in self.conn.execute(
                f"""SELECT asset_id, place_id, 'user' src FROM place_labels WHERE positive=1 AND asset_id IN ({qs})
                    UNION SELECT asset_id, place_id, 'ai' FROM place_ai WHERE asset_id IN ({qs})""", ids + ids):
            places.setdefault(r["asset_id"], {}).setdefault(r["place_id"], r["src"])
        pname = {r["id"]: r["name"] for r in self.conn.execute("SELECT id, name FROM places")}
        tags = {}
        for r in self.conn.execute(
                f"""SELECT asset_id, tag_id, 'user' src FROM tag_labels WHERE positive=1 AND asset_id IN ({qs})
                    UNION SELECT asset_id, tag_id, 'ai' FROM tag_ai WHERE asset_id IN ({qs})""", ids + ids):
            tags.setdefault(r["asset_id"], {}).setdefault(r["tag_id"], r["src"])
        tname = {r["id"]: r["name"] for r in self.conn.execute("SELECT id, name FROM tags")}
        who = {}
        for r in self.conn.execute(
                f"""SELECT f.asset_id, l.person_id, 'user' src FROM face_labels l JOIN faces f ON f.id=l.face_id WHERE l.positive=1 AND f.asset_id IN ({qs})
                    UNION SELECT f.asset_id, i.person_id, 'ai' FROM face_ai i JOIN faces f ON f.id=i.face_id WHERE f.asset_id IN ({qs})""", ids + ids):
            who.setdefault(r["asset_id"], {}).setdefault(r["person_id"], r["src"])
        wname = {r["id"]: r["display_name"] or r["id"] for r in self.conn.execute("SELECT id, display_name FROM persons")}
        nonlive = {r["asset_id"]: r["kind"] for r in self.conn.execute(f"SELECT asset_id, kind FROM asset_nonlive WHERE asset_id IN ({qs})", ids)}
        out = []
        for a in ids:
            i = info.get(a)
            if not i:
                continue
            out.append({"id": a, "kind": i["kind"], "date": _fmt_date(i["date_taken"], i["date_precision"]), "path": paths.get(a, ""),
                        "pets": sorted(pets.get(a, [])), "places": [{"name": pname.get(k, k), "src": v} for k, v in places.get(a, {}).items()] + [{"name": tname.get(k, k), "src": v} for k, v in tags.get(a, {}).items()],
                        "persons": [{"name": wname.get(k, k), "src": v} for k, v in who.get(a, {}).items()], "nonlive": nonlive.get(a), "duration": i["duration_sec"],
                        "sim": round(sim[a], 3) if a in sim else None})
        return out

    def place_list(self) -> list:
        """登録されている場所すべて（見本の枚数・自動で付いた枚数つき）。"""
        c = self.conn
        lab = {r[0]: r[1] for r in c.execute("SELECT place_id, COUNT(*) FROM place_labels WHERE positive=1 GROUP BY place_id")}
        ai = {r[0]: r[1] for r in c.execute("SELECT place_id, COUNT(*) FROM place_ai GROUP BY place_id")}
        from . import placelearn

        return [{"id": r["id"], "name": r["name"], "category": r["category"], "labelled": lab.get(r["id"], 0), "auto": ai.get(r["id"], 0),
                 "thr": placelearn.threshold(c, r["id"])} for r in c.execute("SELECT id, name, category FROM places WHERE active=1 ORDER BY name")]

    def tag_list(self) -> list:
        """場面タグ（kind が scene）の一覧（見本の枚数・自動で付いた枚数つき）。"""
        c = self.conn
        lab = {r[0]: r[1] for r in c.execute("SELECT tag_id, COUNT(*) FROM tag_labels WHERE positive=1 GROUP BY tag_id")}
        ai = {r[0]: r[1] for r in c.execute("SELECT tag_id, COUNT(*) FROM tag_ai GROUP BY tag_id")}
        from . import placelearn

        return [{"id": r["id"], "name": r["name"], "category": r["kind"], "labelled": lab.get(r["id"], 0), "auto": ai.get(r["id"], 0),
                 "thr": placelearn.threshold(c, r["id"], "tag")} for r in c.execute("SELECT id, name, kind FROM tags WHERE active=1 AND kind='scene' ORDER BY name")]

    def facets(self) -> dict:
        c = self.conn
        pets = [{"id": r["id"], "name": r["display_name"] or r["id"], "n": r["n"]} for r in c.execute(
            f"""SELECT pt.id, pt.display_name, COUNT(DISTINCT f.asset_id) n FROM pets pt JOIN file_pets p ON p.pet_id=pt.id
                JOIN files f ON f.id=p.file_id JOIN assets a ON a.id=f.asset_id
                WHERE a.zone='real' AND p.status<>'rejected' AND {CONFIDENT} GROUP BY pt.id ORDER BY n DESC""")]
        years = [{"year": r[0], "n": r[1]} for r in c.execute(
            "SELECT substr(date_taken,1,4) y, COUNT(*) FROM assets WHERE zone='real' AND date_taken IS NOT NULL GROUP BY y ORDER BY y")]
        kinds = {r[0]: r[1] for r in c.execute("SELECT kind, COUNT(*) FROM assets WHERE zone='real' GROUP BY kind")}
        places = [{"id": r[0], "name": r[1], "n": r[2]} for r in c.execute(
            """SELECT pl.id, pl.name, (SELECT COUNT(DISTINCT f.asset_id) FROM file_places x JOIN files f ON f.id=x.file_id WHERE x.place_id=pl.id)
                      + (SELECT COUNT(*) FROM place_labels l WHERE l.place_id=pl.id AND l.positive=1)
                      + (SELECT COUNT(*) FROM place_ai i WHERE i.place_id=pl.id) n
               FROM places pl WHERE pl.active=1 ORDER BY n DESC, pl.name LIMIT 20""")]
        people = [{"id": r[0], "name": r[1], "n": r[2]} for r in c.execute(
            """SELECT p.id, COALESCE(p.display_name, p.id), (SELECT COUNT(DISTINCT f.asset_id) FROM faces f JOIN face_labels l ON l.face_id=f.id WHERE l.person_id=p.id AND l.positive=1)
                      + (SELECT COUNT(DISTINCT f.asset_id) FROM faces f JOIN face_ai i ON i.face_id=f.id WHERE i.person_id=p.id
                         AND f.asset_id NOT IN (SELECT f2.asset_id FROM faces f2 JOIN face_labels l2 ON l2.face_id=f2.id WHERE l2.person_id=p.id AND l2.positive=1)) n
               FROM persons p WHERE p.active=1""") if r[2] > 0]
        scenes = [{"id": r[0], "name": r[1], "n": r[2]} for r in c.execute(
            """SELECT t.id, t.name, (SELECT COUNT(DISTINCT f.asset_id) FROM file_tags x JOIN files f ON f.id=x.file_id WHERE x.tag_id=t.id)
                      + (SELECT COUNT(*) FROM tag_labels l WHERE l.tag_id=t.id AND l.positive=1) + (SELECT COUNT(*) FROM tag_ai i WHERE i.tag_id=t.id) n
               FROM tags t WHERE t.active=1 AND t.kind='scene' ORDER BY n DESC, t.name LIMIT 20""") if r[2] > 0]
        places = [x for x in places if x["n"] > 0]
        extra = {"nonlive": c.execute("SELECT COUNT(DISTINCT asset_id) FROM asset_nonlive").fetchone()[0],
                 "ai": c.execute("SELECT COUNT(*) FROM assets WHERE zone='ai'").fetchone()[0]}
        return {"pets": pets, "years": years, "kinds": kinds, "places": places, "scenes": scenes, "persons": people, "extra": extra}


def _fmt_date(d, prec):
    if not d:
        return ""
    if prec == "year":
        return d[:4] + "年"
    if prec == "month":
        return f"{d[:4]}年{int(d[5:7])}月"
    return f"{d[:4]}年{int(d[5:7])}月{int(d[8:10])}日" if len(d) >= 10 else d[:4] + "年"
