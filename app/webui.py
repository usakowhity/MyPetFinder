"""検索画面のローカルサーバ（標準ライブラリのみ）。自分のPC（127.0.0.1）からだけ開ける。元の写真・動画は読むだけで、変更しない。"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import db, embed, facelearn, petlearn, placelearn
from .report import _mounts, abs_path
from .search import Searcher

WEB = Path(__file__).parent / "web"
THUMB = 320
PLACEHOLDER = ("<svg xmlns='http://www.w3.org/2000/svg' width='320' height='240' viewBox='0 0 320 240'><rect width='320' height='240' fill='#8884'/>"
               "<text x='160' y='128' text-anchor='middle' font-size='22' fill='#888' font-family='sans-serif'>{t}</text></svg>")


class App:
    def __init__(self, cfg, public=False):
        self.cfg = cfg
        self.public = public
        self.job = None
        self.thumbs = cfg.db_path.parent / "thumbs"
        self.lock = threading.Lock()                 # 検索用
        self.flock = threading.Lock()                # 写真の場所の検索用（長い検索に待たされないよう、別の接続を使う）
        self.conn = db.connect(cfg, check_same_thread=False)
        self.fconn = db.connect(cfg, check_same_thread=False)
        self.searcher = Searcher(cfg, self.conn)

    # ---- 場所を付ける
    def place_add(self, name: str, category: str, kind: str = "place", aliases=None):
        name = (name or "").strip()
        bad = '"\\\n\r'
        if not name or len(name) > 40 or any(ch in name for ch in bad):
            raise ValueError("名前が正しくありません（1〜40文字、引用符・改行は使えません）")
        aliases = [a.strip() for a in (aliases or []) if a and a.strip() and len(a.strip()) <= 40 and not any(ch in a for ch in bad)][:20]
        from .textnorm import norm

        table, atable, col = ("places", "place_aliases", "place_id") if kind == "place" else ("tags", "tag_aliases", "tag_id")
        if kind == "place":
            if category not in ("indoor", "semi_outdoor", "outdoor"):
                category = "indoor"
        with self.lock:
            for r in self.conn.execute(f"SELECT id, name FROM {table}"):
                if norm(r["name"]) == norm(name):
                    return {"id": r["id"], "name": r["name"], "existing": True}
            import hashlib
            import re as _re

            base = name.lower() if _re.fullmatch(r"[a-z0-9_]+", name.lower()) else ("p_" if kind == "place" else "s_") + hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
            pid, n = base, 2
            while self.conn.execute(f"SELECT 1 FROM {table} WHERE id=?", (pid,)).fetchone():
                pid, n = f"{base}_{n}", n + 1
            alist = "[" + ", ".join('"' + a + '"' for a in aliases) + "]"
            if kind == "place":
                toml = self.cfg.config_dir / "places.toml"
                text = f'\n[[place]]\nid = "{pid}"\nname = "{name}"\ncategory = "{category}"\naliases = {alist}\n'
            else:
                toml = self.cfg.config_dir / "tags.toml"
                text = f'\n[[tag]]\nid = "{pid}"\nname = "{name}"\nkind = "scene"\naliases = {alist}\n'
            with open(toml, "a", encoding="utf-8", newline="\n") as f:        # 台帳に追記（次回の同期でも残る）
                f.write(text)
            if kind == "place":
                self.conn.execute("INSERT INTO places(id,name,category,active) VALUES(?,?,?,1)", (pid, name, category))
            else:
                self.conn.execute("INSERT INTO tags(id,name,kind,active) VALUES(?,?,'scene',1)", (pid, name))
            for a in [name] + aliases:
                self.conn.execute(f"INSERT OR IGNORE INTO {atable}({col},alias,alias_norm) VALUES(?,?,?)", (pid, a, norm(a)))
            self.conn.commit()
        return {"id": pid, "name": name, "existing": False}

    def place_label(self, ids, pid, mode, kind="place"):
        if mode not in ("add", "neg", "remove"):
            raise ValueError("mode")
        table = "places" if kind == "place" else "tags"
        with self.lock:
            if not self.conn.execute(f"SELECT 1 FROM {table} WHERE id=?", (pid,)).fetchone():
                raise ValueError("登録されていません")
            n = placelearn.label(self.conn, [int(a) for a in ids][:2000], pid, mode, kind)
            return {"n": n, "labelled": len(placelearn._labels(self.conn, pid, 1, kind))}

    def place_suggest(self, pid, limit, offset, kind="place"):
        with self.lock:
            pos, X, ids = self.searcher.matrix()
            cand, total = placelearn.suggest(self.conn, pos, X, ids, pid, limit, offset, kind)
            thr = placelearn.threshold(self.conn, pid, kind)
            items = self.searcher._items([a for a, _ in cand], {a: s for a, s in cand})
            return {"items": items, "total": total, "thr": thr, "labelled": len(placelearn._labels(self.conn, pid, 1, kind))}

    def place_preview(self, pid, thr, kind="place"):
        with self.lock:
            pos, X, ids = self.searcher.matrix()
            return {"count": placelearn.count_at(self.conn, pos, X, ids, pid, thr, kind), "recall": placelearn.loo_recall(self.conn, pos, X, pid, thr, kind)}

    def place_apply(self, pid, thr, kind="place"):
        with self.lock:
            pos, X, ids = self.searcher.matrix()
            return {"count": placelearn.apply(self.conn, pos, X, ids, pid, thr, kind, user=True)}

    def place_clear(self, pid, kind="place"):
        with self.lock:
            return {"cleared": placelearn.clear(self.conn, pid, kind)}

    # ---- 人物と顔
    def faces_matrix(self):
        key = tuple(self.conn.execute("SELECT COUNT(*), COALESCE(MAX(id),0) FROM faces").fetchone())
        if getattr(self, "_fm", None) is None or self._fm[0] != key:
            self._fm = (key,) + facelearn.load(self.conn)
        return self._fm[1:]

    def person_list(self):
        with self.lock:
            c = self.conn
            lab = {r[0]: r[1] for r in c.execute("SELECT person_id, COUNT(*) FROM face_labels WHERE positive=1 GROUP BY person_id")}
            ai = {r[0]: r[1] for r in c.execute("SELECT person_id, COUNT(*) FROM face_ai GROUP BY person_id")}
            nf = c.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            ns = c.execute("SELECT COUNT(*) FROM face_scan WHERE err IS NULL").fetchone()[0]
            return {"faces": nf, "scanned": ns, "persons": [
                {"id": r["id"], "name": r["display_name"] or r["id"], "labelled": lab.get(r["id"], 0), "auto": ai.get(r["id"], 0),
                 "thr": facelearn.threshold(c, r["id"])} for r in c.execute("SELECT id, display_name FROM persons WHERE active=1 ORDER BY display_name")]}

    def person_add(self, name, aliases):
        name = (name or "").strip()
        bad = '"\\\n\r'
        if not name or len(name) > 40 or any(ch in name for ch in bad):
            raise ValueError("名前が正しくありません（1〜40文字、引用符・改行は使えません）")
        aliases = [a.strip() for a in (aliases or []) if a and a.strip() and len(a.strip()) <= 40 and not any(ch in a for ch in bad)][:20]
        from .textnorm import norm

        with self.lock:
            for r in self.conn.execute("SELECT id, display_name FROM persons"):
                if norm(r["display_name"] or r["id"]) == norm(name):
                    return {"id": r["id"], "name": r["display_name"], "existing": True}
            import hashlib

            pid, n = "h_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 2
            base = pid
            while self.conn.execute("SELECT 1 FROM persons WHERE id=?", (pid,)).fetchone():
                pid, n = f"{base}_{n}", n + 1
            alist = "[" + ", ".join('"' + a + '"' for a in [name] + [a for a in aliases if a != name]) + "]"
            with open(self.cfg.config_dir / "persons.toml", "a", encoding="utf-8", newline="\n") as f:
                f.write(f'\n[[person]]\nid = "{pid}"\ndisplay_name = "{name}"\nis_owner = false\naliases = {alist}\nnotes = ""\n')
            self.conn.execute("INSERT INTO persons(id,display_name,is_owner,active) VALUES(?,?,0,1)", (pid, name))
            for a in [name] + aliases:
                self.conn.execute("INSERT OR IGNORE INTO person_aliases(person_id,alias,alias_norm) VALUES(?,?,?)", (pid, a, norm(a)))
            self.conn.commit()
        return {"id": pid, "name": name, "existing": False}

    def _person_ok(self, pid):
        if not self.conn.execute("SELECT 1 FROM persons WHERE id=?", (pid,)).fetchone():
            raise ValueError("その人物は登録されていません")

    def face_label(self, ids, pid, mode):
        if mode not in ("add", "neg", "remove"):
            raise ValueError("mode")
        with self.lock:
            self._person_ok(pid)
            n = facelearn.label(self.conn, [int(a) for a in ids][:5000], pid, mode)
            return {"n": n, "labelled": len(facelearn._labels(self.conn, pid, 1))}

    # ---- ペットに名前を付ける（人物と同じ操作。特徴は DINOv2）
    def pets_matrix(self):
        key = (self.conn.execute("SELECT COUNT(*), COALESCE(MAX(asset_id),0) FROM embeddings").fetchone()[0:2], embed.pet_model(self.conn))
        if getattr(self, "_pm", None) is None or self._pm[0] != key:
            self._pm = (key,) + petlearn.load(self.conn)
        return self._pm[1:]

    def pet_list(self):
        with self.lock:
            from . import finder

            return {"model": embed.pet_model(self.conn), "pets": petlearn.pet_list(self.conn), "warnings": finder.load_warnings(self.conn),
                    "ignored": self.conn.execute("SELECT COUNT(*) FROM pet_ignore").fetchone()[0]}

    @staticmethod
    def split_aliases(aliases):
        """別の呼び名: 「、」「,」「;」「/」「・」空白のどれで区切ってもよい。リストでも文字列でも受ける。"""
        if isinstance(aliases, str):
            aliases = [aliases]
        out = []
        for a in aliases or []:
            out += [x for x in re.split(r"[,、，;；/／・\s]+", str(a)) if x]
        return list(dict.fromkeys(out))

    def pet_alias(self, pid, aliases):
        """既存のペットの別の呼び名を、置き換える（pets.toml も書き換える）。"""
        from .textnorm import norm

        bad = '"\\\n\r'
        als = [a for a in self.split_aliases(aliases) if len(a) <= 40 and not any(ch in a for ch in bad)][:20]
        with self.lock:
            self._pet_ok(pid)
            name = self.conn.execute("SELECT display_name FROM pets WHERE id=?", (pid,)).fetchone()[0] or pid
            full = [name] + [a for a in als if a != name]
            path = self.cfg.config_dir / "pets.toml"
            txt = path.read_text(encoding="utf-8") if path.exists() else ""
            blocks = re.split(r"(?m)^(?=\[\[pet\]\])", txt)
            line = "aliases = [" + ", ".join('"' + a + '"' for a in full) + "]"
            done = False
            for i, b in enumerate(blocks):
                if b.startswith("[[pet]]") and re.search(r'(?m)^id\s*=\s*"' + re.escape(pid) + '"', b):
                    b = re.sub(r"(?m)^aliases\s*=.*$", line, b) if re.search(r"(?m)^aliases\s*=", b) else b.rstrip("\n") + "\n" + line + "\n"
                    blocks[i], done = b, True
            if not done:                                        # 台帳に無い（個人用の台帳に書いていない）ときは、追記する
                blocks.append(f'\n[[pet]]\nid = "{pid}"\ndisplay_name = "{name}"\n{line}\n')
            path.write_text("".join(blocks), encoding="utf-8", newline="\n")
            self.conn.execute("DELETE FROM pet_aliases WHERE pet_id=?", (pid,))
            for a in full:
                self.conn.execute("INSERT OR IGNORE INTO pet_aliases(pet_id,alias,alias_norm) VALUES(?,?,?)", (pid, a, norm(a)))
            self.conn.commit()
        return {"aliases": als}

    def pet_add(self, name, species, aliases, start, end):
        aliases = self.split_aliases(aliases)
        name = (name or "").strip()
        bad = '"\\\n\r'
        if not name or len(name) > 40 or any(ch in name for ch in bad):
            raise ValueError("名前が正しくありません（1〜40文字、引用符・改行は使えません）")
        species = (species or "").strip()[:20]
        if any(ch in species for ch in bad):
            raise ValueError("種類が正しくありません")
        aliases = [a.strip() for a in (aliases or []) if a and a.strip() and len(a.strip()) <= 40 and not any(ch in a for ch in bad)][:20]
        dates = {}
        for k, v in (("start_date", start), ("end_date", end)):
            v = (v or "").strip()
            if v:
                if not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", v):
                    raise ValueError("日付は 2015 または 2015-04-01 の形で入力してください")
                dates[k] = v
        from .textnorm import norm

        with self.lock:
            for r in self.conn.execute("SELECT id, display_name FROM pets"):
                if norm(r["display_name"] or r["id"]) == norm(name):
                    return {"id": r["id"], "name": r["display_name"], "existing": True}
            import hashlib

            pid, n = "p_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 2
            base = pid
            while self.conn.execute("SELECT 1 FROM pets WHERE id=?", (pid,)).fetchone():
                pid, n = f"{base}_{n}", n + 1
            alist = "[" + ", ".join('"' + a + '"' for a in [name] + [a for a in aliases if a != name]) + "]"
            extra = "".join(f'{k} = "{v}"\n' for k, v in dates.items())
            with open(self.cfg.config_dir / "pets.toml", "a", encoding="utf-8", newline="\n") as f:
                f.write(f'\n[[pet]]\nid = "{pid}"\ndisplay_name = "{name}"\nspecies = "{species}"\naliases = {alist}\n{extra}')
            self.conn.execute("INSERT INTO pets(id,display_name,species,active,start_date,end_date) VALUES(?,?,?,1,?,?)",
                              (pid, name, species or None, dates.get("start_date"), dates.get("end_date")))
            for a in [name] + aliases:
                self.conn.execute("INSERT OR IGNORE INTO pet_aliases(pet_id,alias,alias_norm) VALUES(?,?,?)", (pid, a, norm(a)))
            self.conn.commit()
        return {"id": pid, "name": name, "existing": False}

    def _pet_ok(self, pid):
        if not self.conn.execute("SELECT 1 FROM pets WHERE id=?", (pid,)).fetchone():
            raise ValueError("そのペットは登録されていません")

    def pet_clusters(self, thr=None):
        with self.lock:
            ids, X = self.pets_matrix()
            key = (len(ids), self.conn.execute("SELECT COUNT(*) FROM file_pets WHERE origin='user'").fetchone()[0],
                   self.conn.execute("SELECT COUNT(*) FROM pet_ignore").fetchone()[0], thr)
            if getattr(self, "_pc", None) is None or self._pc[0] != key:
                self._pc = (key, petlearn.groups(self.conn, ids, X, thr))
            return {"groups": self._pc[1], "model": embed.pet_model(self.conn)}

    def pet_items(self, pairs):
        return [{"asset_id": a, "sim": round(s, 3)} for a, s in pairs]

    def pet_suggest(self, pid, limit, offset):
        with self.lock:
            self._pet_ok(pid)
            ids, X = self.pets_matrix()
            cand, total = petlearn.suggest(self.conn, ids, X, pid, limit, offset)
            return {"items": self.pet_items(cand), "total": total, "labelled": len(petlearn.examples(self.conn, ids, pid)[0])}

    def pet_review(self, pid, kind, limit, offset):
        with self.lock:
            self._pet_ok(pid)
            ids, X = self.pets_matrix()
            cand, total = petlearn.review(self.conn, ids, X, pid, "ai" if kind == "ai" else "human", limit, offset)
            return {"items": self.pet_items(cand), "total": total}

    def pet_label(self, ids, pid, mode, old=None):
        if mode not in ("add", "neg", "remove", "move"):
            raise ValueError("mode")
        with self.lock:
            self._pet_ok(pid)
            ids = [int(a) for a in ids][:5000]
            if mode == "move":
                self._pet_ok(old or "")
                n = petlearn.move(self.conn, ids, old, pid)
            else:
                n = petlearn.label(self.conn, ids, pid, mode)
            self._pc = None
            lab = [p for p in petlearn.pet_list(self.conn) if p["id"] == pid]
            return {"n": n, "labelled": lab[0]["labelled"] if lab else 0}

    def pet_ignore(self, ids, on=True):
        with self.lock:
            n = petlearn.ignore(self.conn, [int(a) for a in ids][:20000], on)
            self._pc = None
            return {"n": n}

    def pet_identify(self):
        """判定のやり直し（メニュー11と同じ）を、裏で実行する。"""
        from . import identify

        if self.public:
            from . import finder

            self.job = self.job or finder.Job()
            finder.identify_job(self.cfg, self.job)
            return {"started": True}
        with self.lock:
            if getattr(self, "_pjob", None) and self._pjob.get("running"):
                raise ValueError("判定中です。終わるまでお待ちください。")
            job = {"running": True, "log": [], "error": None}
            self._pjob = job

        def run():
            conn = db.connect(self.cfg, check_same_thread=False)
            try:
                identify.run(self.cfg, conn, log=lambda m: job["log"].append(str(m)))
            except Exception as ex:                                  # noqa: BLE001
                job["error"] = f"{type(ex).__name__}: {ex}"
            finally:
                job["running"] = False
                conn.close()
        threading.Thread(target=run, daemon=True).start()
        return {"started": True}

    def pet_job(self):
        if self.public and self.job:
            j = self.job.snapshot()
            return {"running": j["running"], "error": j["error"], "log": j["log"]}
        j = getattr(self, "_pjob", None)
        return {"idle": True} if not j else {"running": j["running"], "error": j["error"], "log": j["log"][-8:]}

    # ---- アルバム書き出し（バックグラウンドで実行し、進み具合を返す）
    def album_start(self, body):
        from . import album

        with self.lock:
            if getattr(self, "_job", None) and self._job.get("running"):
                raise ValueError("書き出し中です。終わるまでお待ちください。")
            ids = [int(a) for a in body.get("asset_ids", [])][:album.MAX_ITEMS]
            if not ids:
                o = body.get("opts", {})
                opts = {"include_ai": bool(o.get("ai")), "loose": bool(o.get("loose")), "show_nonlive": bool(o.get("nonlive")), "oldest": bool(o.get("old"))}
                res = self.searcher.search(str(body.get("q", "")), opts, limit=album.MAX_ITEMS, offset=0, ids_only=True)
                ids = res["ids"]
            if not ids:
                raise ValueError("書き出す写真・動画がありません。")
            dest = album._unique_dir(album.check_dest(self.cfg, str(body.get("dest", "")), str(body.get("name", ""))))
            mode = "link" if body.get("mode") == "link" else "copy"
        job = {"running": True, "n": 0, "total": len(ids), "cancel": False, "result": None, "error": None, "dir": str(dest)}
        self._job = job

        def run():
            conn = db.connect(self.cfg, check_same_thread=False)
            try:
                job["result"] = album.export(conn, ids, dest, mode, progress=lambda n, t, nm: job.update(n=n, total=t), cancel=lambda: job["cancel"])
            except Exception as ex:                                  # noqa: BLE001
                job["error"] = f"{type(ex).__name__}: {ex}"
            finally:
                job["running"] = False
                conn.close()
        threading.Thread(target=run, daemon=True).start()
        return {"total": len(ids), "dir": str(dest)}

    def finder_status(self):
        from . import finder

        with self.lock:
            st = finder.status(self.conn)
        j = (self.job or finder.Job()).snapshot()
        return {"public": self.public, **st, "job": j}

    def finder_analyze(self):
        from . import finder

        self.job = self.job or finder.Job()
        self.job.start(self.cfg)
        return {"started": True}

    def album_status(self, cancel=False):
        j = getattr(self, "_job", None)
        if not j:
            return {"idle": True}
        if cancel:
            j["cancel"] = True
        return {k: j[k] for k in ("running", "n", "total", "result", "error", "dir")}

    def face_ignore(self, ids, on=True):
        with self.lock:
            n = facelearn.ignore(self.conn, [int(a) for a in ids][:20000], on)
            self._cl = None
            return {"n": n, "ignored": self.conn.execute("SELECT COUNT(*) FROM face_ignore").fetchone()[0]}

    def _face_items(self, pairs):
        if not pairs:
            return []
        qs = ",".join("?" * len(pairs))
        a = {r["id"]: r["asset_id"] for r in self.conn.execute(f"SELECT id, asset_id FROM faces WHERE id IN ({qs})", [f for f, _ in pairs])}
        return [{"id": f, "asset_id": a.get(f), "sim": round(s, 3) if s is not None else None} for f, s in pairs]

    def face_suggest(self, pid, limit, offset):
        with self.lock:
            ids, _, F = self.faces_matrix()
            cand, total = facelearn.suggest(self.conn, ids, F, pid, limit, offset)
            return {"items": self._face_items(cand), "total": total, "thr": facelearn.threshold(self.conn, pid),
                    "labelled": len(facelearn._labels(self.conn, pid, 1))}

    def face_review(self, pid, kind, limit, offset):
        with self.lock:
            ids, _, F = self.faces_matrix()
            rows, total = facelearn.review(self.conn, ids, F, pid, "ai" if kind == "ai" else "human", limit, offset)
            return {"items": self._face_items(rows), "total": total, "kind": "ai" if kind == "ai" else "human",
                    "labelled": len(facelearn._labels(self.conn, pid, 1))}

    def face_preview(self, pid, thr):
        with self.lock:
            ids, _, F = self.faces_matrix()
            return {"count": facelearn.count_at(self.conn, ids, F, pid, thr), "recall": facelearn.loo_recall(self.conn, ids, F, pid, thr)}

    def face_apply(self, pid, thr):
        with self.lock:
            self._person_ok(pid)
            facelearn.set_threshold(self.conn, pid, thr)
            ids, _, F = self.faces_matrix()
            cnt = facelearn.apply_all(self.conn, ids, F, lambda p: facelearn.threshold(self.conn, p))
            return {"count": cnt.get(pid, 0), "all": cnt}

    def face_clusters(self, refresh=False):
        with self.lock:
            key = tuple(self.conn.execute("SELECT COUNT(*), COALESCE(MAX(id),0) FROM faces").fetchone()) + (self.conn.execute("SELECT COUNT(*) FROM face_labels").fetchone()[0], self.conn.execute("SELECT COUNT(*) FROM face_ignore").fetchone()[0])
            if refresh or getattr(self, "_cl", None) is None or self._cl[0] != key:
                ids, _, F = self.faces_matrix()
                self._cl = (key, facelearn.clusters(self.conn, ids, F))
            return {"clusters": self._cl[1], "faces": key[0], "ignored": key[3]}

    @staticmethod
    def _video_thumb(src, cache):
        try:
            import cv2

            cap = cv2.VideoCapture(str(src))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                if fps > 1e-3 and cnt > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(1.0, cnt / fps / 2) * fps))
                ok, img = cap.read()
            finally:
                cap.release()
            if ok and img is not None:
                h, w = img.shape[:2]
                s = THUMB / max(h, w)
                img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(cache), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception:
            pass

    def face_crop(self, face_id: int):
        cache = self.thumbs.parent / "faces" / f"{face_id}.jpg"
        if cache.exists():
            return cache.read_bytes(), "image/jpeg"
        with self.flock:
            r = self.fconn.execute("SELECT asset_id, x, y, w, h FROM faces WHERE id=?", (face_id,)).fetchone()
        src = self.file_of(r["asset_id"]) if r else None
        if src is None:
            return PLACEHOLDER.format(t="顔").encode(), "image/svg+xml"
        try:
            from PIL import Image, ImageOps

            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                W, H = im.size
                m = 0.35
                x0, y0 = (r["x"] - m * r["w"]) * W, (r["y"] - m * r["h"]) * H
                x1, y1 = (r["x"] + (1 + m) * r["w"]) * W, (r["y"] + (1 + m) * r["h"]) * H
                cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
                box = (max(cx - side / 2, 0), max(cy - side / 2, 0), min(cx + side / 2, W), min(cy + side / 2, H))
                crop = im.crop(tuple(int(v) for v in box)).resize((112, 112))
                cache.parent.mkdir(parents=True, exist_ok=True)
                crop.save(cache, "JPEG", quality=85)
            return cache.read_bytes(), "image/jpeg"
        except Exception:
            return PLACEHOLDER.format(t="顔").encode(), "image/svg+xml"

    def file_of(self, asset_id: int):
        with self.flock:
            rows = self.fconn.execute("SELECT vol_id, volpath FROM files WHERE asset_id=? AND status='present' ORDER BY id DESC", (asset_id,)).fetchall()
            mounts = _mounts(self.fconn)
        for r in rows:
            p = Path(abs_path(mounts, r["vol_id"], r["volpath"]))
            if p.is_file():
                return p
        return None

    def kind_of(self, asset_id: int):
        with self.flock:
            r = self.fconn.execute("SELECT kind FROM assets WHERE id=?", (asset_id,)).fetchone()
        return r["kind"] if r else None

    def thumb(self, asset_id: int):
        """(バイト列, content-type)。作れないときは、説明の絵を返す。"""
        cache = self.thumbs / f"{asset_id}.jpg"
        if cache.exists():
            return cache.read_bytes(), "image/jpeg"
        src = self.file_of(asset_id)
        kind = self.kind_of(asset_id)
        if src is None:
            return PLACEHOLDER.format(t="ファイルが見つかりません").encode(), "image/svg+xml"
        try:
            self.thumbs.mkdir(parents=True, exist_ok=True)
            if kind == "video":
                ff = shutil.which("ffmpeg")
                if ff:
                    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "1", "-i", str(src), "-frames:v", "1", "-vf", f"scale={THUMB}:-2", str(cache)],
                                   timeout=30, check=False)
                if not cache.exists():
                    self._video_thumb(src, cache)                  # ffmpeg がなくても、OpenCV で動画の1コマを取り出す
                if cache.exists():
                    return cache.read_bytes(), "image/jpeg"
                return PLACEHOLDER.format(t="動画").encode(), "image/svg+xml"
            from PIL import Image, ImageOps

            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((THUMB, THUMB))
                im.convert("RGB").save(cache, "JPEG", quality=80)
            return cache.read_bytes(), "image/jpeg"
        except Exception:
            return PLACEHOLDER.format(t="表示できません").encode(), "image/svg+xml"


def make_handler(app: App, port_holder: dict):
    class H(BaseHTTPRequestHandler):
        server_version = "MyPetDatabase"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="text/html; charset=utf-8", extra=None):
            try:
                self._send0(code, body, ctype, extra)
            except (ConnectionError, TimeoutError):
                pass          # ブラウザが先に接続を閉じた（画面の切り替え・再検索）。エラーではない

        def _send0(self, code, body: bytes, ctype, extra):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith(("text/html", "application/json")) else "max-age=3600")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self):
            host = (self.headers.get("Host") or "").split(":")[0].lower()
            return host in ("127.0.0.1", "localhost")     # 他のサイトからの読み出し（DNSリバインディング）を防ぐ

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._send(200, (WEB / "index.html").read_bytes())
                if u.path == "/api/facets":
                    with app.lock:
                        return self._json(app.searcher.facets())
                if u.path == "/api/search":
                    opts = {"include_ai": q.get("ai") == "1", "loose": q.get("loose") == "1", "show_nonlive": q.get("nonlive") == "1", "oldest": q.get("order") == "old"}
                    t0 = time.perf_counter()
                    with app.lock:
                        res = (app.searcher.search(q.get("q", ""), opts, limit=min(int(q.get("limit", 60)), 200), offset=int(q.get("offset", 0))))
                    print(f"  検索 {(time.perf_counter() - t0) * 1000:,.0f} ms  {q.get('q', '')[:40]}  → {res['total']:,} 件")
                    return self._json(res)
                kind = "tag" if q.get("type") == "tag" else "place"
                if u.path == "/api/places":
                    with app.lock:
                        return self._json(app.searcher.tag_list() if kind == "tag" else app.searcher.place_list())
                if u.path == "/api/place/suggest":
                    return self._json(app.place_suggest(q.get("place", ""), min(int(q.get("limit", 60)), 200), int(q.get("offset", 0)), kind))
                if u.path == "/api/place/preview":
                    return self._json(app.place_preview(q.get("place", ""), float(q.get("thr", placelearn.DEFAULT_THR)), kind))
                if u.path == "/api/persons":
                    return self._json(app.person_list())
                if u.path == "/api/face/suggest":
                    return self._json(app.face_suggest(q.get("person", ""), min(int(q.get("limit", 60)), 300), int(q.get("offset", 0))))
                if u.path == "/api/face/review":
                    return self._json(app.face_review(q.get("person", ""), q.get("kind", "human"), min(int(q.get("limit", 60)), 300), int(q.get("offset", 0))))
                if u.path == "/api/face/preview":
                    return self._json(app.face_preview(q.get("person", ""), float(q.get("thr", facelearn.DEFAULT_THR))))
                if u.path == "/api/mode":
                    return self._json({"public": app.public})
                if u.path == "/api/finder/status":
                    return self._json(app.finder_status())
                if u.path == "/api/petlist":
                    return self._json(app.pet_list())
                if u.path == "/api/pet/clusters":
                    return self._json(app.pet_clusters(float(q["thr"]) if q.get("thr") else None))
                if u.path == "/api/pet/suggest":
                    return self._json(app.pet_suggest(q.get("pet", ""), min(int(q.get("limit", 60)), 300), int(q.get("offset", 0))))
                if u.path == "/api/pet/review":
                    return self._json(app.pet_review(q.get("pet", ""), q.get("kind", "human"), min(int(q.get("limit", 60)), 300), int(q.get("offset", 0))))
                if u.path == "/api/pet/job":
                    return self._json(app.pet_job())
                if u.path == "/api/pet/ignored":
                    with app.lock:
                        return self._json({"ids": [r[0] for r in app.conn.execute("SELECT asset_id FROM pet_ignore")]})
                if u.path == "/api/album/status":
                    return self._json(app.album_status())
                if u.path == "/api/face/ignored":
                    with app.lock:
                        return self._json({"ids": [r[0] for r in app.conn.execute("SELECT face_id FROM face_ignore")]})
                if u.path == "/api/face/clusters":
                    return self._json(app.face_clusters(q.get("refresh") == "1"))
                m = re.fullmatch(r"/face/(\d+)", u.path)
                if m:
                    body, ct = app.face_crop(int(m.group(1)))
                    return self._send(200, body, ct)
                if u.path == "/api/parse":
                    with app.lock:
                        return self._json(app.searcher.parse(q.get("q", "")))
                m = re.fullmatch(r"/thumb/(\d+)", u.path)
                if m:
                    body, ct = app.thumb(int(m.group(1)))
                    return self._send(200, body, ct)
                m = re.fullmatch(r"/file/(\d+)", u.path)
                if m:
                    return self._file(int(m.group(1)))
                return self._send(404, b"not found", "text/plain")
            except Exception as ex:
                return self._json({"error": f"{type(ex).__name__}: {ex}"}, 500)

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1].split(":")[0].lower() not in ("127.0.0.1", "localhost"):
                return self._send(403, b"forbidden", "text/plain")                 # 他のサイトからの書き込みを防ぐ
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(415, b"json only", "text/plain")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(n, 1 << 20)).decode("utf-8") or "{}")
                path = urlparse(self.path).path
                kind = "tag" if body.get("type") == "tag" else "place"
                if path == "/api/person/add":
                    return self._json(app.person_add(body.get("name", ""), body.get("aliases")))
                if path == "/api/face/label":
                    return self._json(app.face_label(body.get("face_ids", []), body.get("person_id", ""), body.get("mode", "add")))
                if path == "/api/finder/analyze":
                    return self._json(app.finder_analyze())
                if path == "/api/pet/alias":
                    return self._json(app.pet_alias(body.get("pet_id", ""), body.get("aliases", [])))
                if path == "/api/pet/add":
                    return self._json(app.pet_add(body.get("name", ""), body.get("species", ""), body.get("aliases"), body.get("start"), body.get("end")))
                if path == "/api/pet/label":
                    return self._json(app.pet_label(body.get("asset_ids", []), body.get("pet_id", ""), body.get("mode", "add"), body.get("old")))
                if path == "/api/pet/ignore":
                    return self._json(app.pet_ignore(body.get("asset_ids", []), bool(body.get("on", True))))
                if path == "/api/pet/identify":
                    return self._json(app.pet_identify())
                if path == "/api/place/clear":
                    return self._json(app.place_clear(body.get("place_id", ""), "tag" if body.get("type") == "tag" else "place"))
                if path == "/api/album/start":
                    return self._json(app.album_start(body))
                if path == "/api/album/cancel":
                    return self._json(app.album_status(cancel=True))
                if path == "/api/face/ignore":
                    return self._json(app.face_ignore(body.get("face_ids", []), bool(body.get("on", True))))
                if path == "/api/face/apply":
                    return self._json(app.face_apply(body.get("person_id", ""), float(body.get("thr", facelearn.DEFAULT_THR))))
                if path == "/api/place/add":
                    return self._json(app.place_add(body.get("name", ""), body.get("category", "indoor"), kind, body.get("aliases")))
                if path == "/api/place/label":
                    return self._json(app.place_label(body.get("asset_ids", []), body.get("place_id", ""), body.get("mode", "add"), kind))
                if path == "/api/place/apply":
                    return self._json(app.place_apply(body.get("place_id", ""), float(body.get("thr", placelearn.DEFAULT_THR)), kind))
                return self._send(404, b"not found", "text/plain")
            except ValueError as ex:
                return self._json({"error": str(ex)}, 400)
            except Exception as ex:
                return self._json({"error": f"{type(ex).__name__}: {ex}"}, 500)

        do_HEAD = do_GET

        def _file(self, asset_id):
            p = app.file_of(asset_id)
            if p is None:
                return self._send(404, b"file not found", "text/plain")
            size = p.stat().st_size
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            code = 200
            if rng:
                m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else end
                    elif m.group(2):
                        start = max(size - int(m.group(2)), 0)
                    if start > end or start >= size:
                        return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                    end = min(end, size - 1)
                    code = 206
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(p, "rb") as f:               # 読み取り専用で開く
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def serve(cfg, port=8765, open_browser=True, log=print, public=False) -> int:
    app = App(cfg, public=public)
    holder = {}
    srv = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), make_handler(app, holder))
            break
        except OSError:
            continue
    if srv is None:
        log("使える番号（ポート）が見つかりません。")
        return 2
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    threading.Thread(target=app.searcher.warm, daemon=True).start()
    log(f"検索画面: {url}  （終了するには、この黒い画面で Ctrl+C を押すか、閉じてください）")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    srv.handle_error = lambda request, client_address: None      # 接続が途中で閉じられた時の長いエラー表示を出さない
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
