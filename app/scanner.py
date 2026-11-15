"""フォルダの走査（preview）と目録作成（scan）。

- 元の写真・動画は読み取るだけで、変更・移動・削除は一切しない。
- 2回目以降は、サイズと更新日時が同じファイルを再計算せず飛ばす（途中で止めても再開できる）。
- 場所は「ボリュームID + ボリューム内の相対パス」で記録するので、ドライブ文字が変わっても
  同じファイルとして認識する。
"""
from __future__ import annotations

import datetime as dt
import fnmatch
import os
import sys
import time
from collections import Counter, defaultdict

from . import dates, media, volumes
from .config import Settings
from .db import now_iso
from .labels import LabelIndex, store_labels
from .pethints import PetIndex, store_hints
from .textnorm import norm

BATCH_COMMIT = 500


# ---------------------------------------------------------------- 補助
def fs(path: str) -> str:
    """Windowsの長いパス（約260文字超）に対応する接頭辞を付ける。"""
    if sys.platform == "win32" and len(path) >= 240 and not path.startswith("\\\\?\\"):
        ap = os.path.abspath(path)
        if ap.startswith("\\\\"):
            return "\\\\?\\UNC\\" + ap[2:]
        return "\\\\?\\" + ap
    return path


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _is_junction(entry) -> bool:
    f = getattr(entry, "is_junction", None)
    try:
        return bool(f()) if f else False
    except OSError:
        return False


def sorted_roots(cfg: Settings) -> list[dict]:
    """深い（具体的な）フォルダを先に処理する。重なりがあるとき、その zone が優先される。"""
    return sorted(
        cfg.scans, key=lambda r: -len(os.path.normcase(os.path.abspath(r["path"])))
    )


def walk_media(root: str, cfg: Settings, on_error):
    """(フルパス, ファイル名, stat) を順に返す。シンボリックリンク等は辿らない。"""
    app_norm = os.path.normcase(os.path.abspath(cfg.app_root))
    exts = cfg.media_exts
    stack = [root]
    while stack:
        d = stack.pop()
        dn = os.path.normcase(os.path.abspath(d))
        if dn == app_norm or dn.startswith(app_norm + os.sep):
            continue  # MyPetDatabase 自身は走査しない
        try:
            with os.scandir(fs(d)) as it:
                entries = sorted(it, key=lambda e: e.name.lower())
        except OSError as ex:
            on_error(d, ex)
            continue
        subdirs = []
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    nm = e.name.lower()
                    if (
                        _is_junction(e)
                        or nm in cfg.exclude_dir_names
                        or any(fnmatch.fnmatchcase(nm, pat) for pat in cfg.exclude_dir_patterns)
                    ):
                        continue
                    subdirs.append(os.path.join(d, e.name))
                elif e.is_file(follow_symlinks=False):
                    ext = os.path.splitext(e.name)[1].lower()
                    if ext in exts:
                        yield os.path.join(d, e.name), e.name, e.stat(follow_symlinks=False)
            except OSError as ex:
                on_error(os.path.join(d, e.name), ex)
        stack.extend(reversed(subdirs))


# ---------------------------------------------------------------- preview
def run_preview(cfg: Settings, log=print) -> None:
    """ハッシュを計算せず、件数と容量だけを数える（フォルダ構成の確認用）。"""
    log("=== プレビュー（ファイルは読まず、件数と容量だけ数えます） ===")
    grand_n, grand_size = 0, 0
    counted: set = set()  # 重なった走査先で二重に数えない（scan と同じ扱い）
    for root in sorted_roots(cfg):
        path = root["path"]
        log(f"\n■ {path}   [区分: {root['zone']}]")
        if not os.path.isdir(path):
            log("  ※ フォルダが見つかりません（ドライブ未接続？ パスの誤り？）。飛ばします。")
            continue
        by_kind = defaultdict(lambda: [0, 0])
        by_ext: Counter = Counter()
        by_top = defaultdict(lambda: [0, 0])
        errors = []
        t0 = time.time()
        for full, name, st in walk_media(path, cfg, lambda p, ex: errors.append((p, ex))):
            ck = os.path.normcase(os.path.abspath(full))
            if ck in counted:
                continue
            counted.add(ck)
            ext = os.path.splitext(name)[1].lower()
            kind = "image" if ext in cfg.image_exts else "video"
            by_kind[kind][0] += 1
            by_kind[kind][1] += st.st_size
            by_ext[ext] += 1
            parts = os.path.relpath(full, path).split(os.sep)
            top = parts[0] if len(parts) > 1 else "(直下)"
            by_top[top][0] += 1
            by_top[top][1] += st.st_size
        n = sum(v[0] for v in by_kind.values())
        size = sum(v[1] for v in by_kind.values())
        grand_n += n
        grand_size += size
        log(
            f"  画像 {by_kind['image'][0]:,} 件 ({human_size(by_kind['image'][1])}) / "
            f"動画 {by_kind['video'][0]:,} 件 ({human_size(by_kind['video'][1])})   "
            f"[{time.time() - t0:.0f}秒]"
        )
        if by_ext:
            log("  拡張子: " + ", ".join(f"{e} {c:,}" for e, c in by_ext.most_common(8)))
        if len(by_top) > 1:
            log("  直下のフォルダ別（容量の大きい順・上位15）:")
            for top, (c, s) in sorted(by_top.items(), key=lambda kv: -kv[1][1])[:15]:
                log(f"    {top}: {c:,} 件 / {human_size(s)}")
        if errors:
            log(f"  ※ 読めなかった場所が {len(errors)} 件あります（例: {errors[0][0]}）")
    log(f"\n合計: {grand_n:,} 件 / {human_size(grand_size)}")
    log("不要なフォルダがあれば config/settings.toml の exclude_dir_names に追加してください。")


# ---------------------------------------------------------------- scan
def _zone_for(root_zone: str, volpath: str, cfg: Settings) -> str:
    if root_zone in ("real", "ai"):
        return root_zone
    p = norm(volpath)
    return "ai" if any(k and k in p for k in cfg.ai_keywords) else "unknown"


def _mtime_dt(st):
    try:
        d = dt.datetime.fromtimestamp(st.st_mtime_ns / 1e9)
    except (OverflowError, OSError, ValueError):
        return None
    return d if media.MIN_YEAR <= d.year <= dt.date.today().year + 1 else None


def _process_file(full, name, st, volpath, zone, pets, cfg, known_hash=None, labels=None):
    ext = os.path.splitext(name)[1].lower()
    kind = "image" if ext in cfg.image_exts else "video"
    fp = fs(full)
    hash_key = known_hash or media.file_hash(fp, cfg.hash_mode)
    meta = media.read_meta(fp, ext)

    cands = dates.path_candidates(volpath, name)
    if meta.get("exif_dt"):
        cands.append({"source": meta.get("exif_kind", "exif"), "precision": "day", "date": meta["exif_dt"]})
    if meta.get("container_dt"):
        cands.append({"source": "container", "precision": "day", "date": meta["container_dt"]})
    md = _mtime_dt(st)
    if md:
        cands.append({"source": "mtime", "precision": "day", "date": md})
    d = dates.choose(cands)

    # フォルダ名に含まれるペット名は、実写だけの手掛かりにする
    # （AI生成画像のフォルダ名 Pochi / Sample などを、実在のペットと取り違えないため）
    # 名前が一致するだけでなく、撮影年がそのペットの生存期間内かも見て、手掛かりの強さを決める
    pet_hints = pets.hints(volpath, zone, d["date_taken"], d["date_confidence"], d["date_source"], d["date_conflict"])
    hints = [h["pet_id"] for h in pet_hints]
    return {
        "ext": ext,
        "kind": kind,
        "hash_key": hash_key,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "duration": meta.get("duration"),
        "zone": zone,
        "hints": ",".join(hints) if hints else None,
        "pet_hints": pet_hints,
        "labels": labels.match(volpath) if labels else {"tags": {}, "places": {}},
        "camera_make": meta.get("camera_make"),
        "camera_model": meta.get("camera_model"),
        **d,
    }


def run_scan(cfg: Settings, conn, log=print, force_rehash: bool = False, only: str | None = None,
             redate: bool = False) -> dict:
    started = now_iso()
    cur = conn.execute(
        "INSERT INTO scan_runs(started, status, hash_mode) VALUES(?, 'running', ?)",
        (started, cfg.hash_mode),
    )
    run_id = cur.lastrowid
    conn.commit()

    pets = PetIndex(conn)
    labels = LabelIndex(conn)
    stats = Counter()
    seen: set = set()
    interrupted = False
    t0 = time.time()
    last_log = t0
    pending = 0

    def log_error(path, ex):
        stats["errors"] += 1
        conn.execute(
            "INSERT INTO scan_errors(run_id, path, message) VALUES(?,?,?)", (run_id, str(path), str(ex))
        )

    try:
        for root in sorted_roots(cfg):
            path = root["path"]
            if only and norm(only) not in norm(path):
                continue
            if not os.path.isdir(path):
                log(f"※ 見つかりません（飛ばします）: {path}")
                stats["roots_missing"] += 1
                continue
            vol = volumes.volume_of(path)
            conn.execute(
                """INSERT INTO volumes(vol_id, label, last_mount, last_seen) VALUES(?,?,?,?)
                   ON CONFLICT(vol_id) DO UPDATE SET label=excluded.label,
                     last_mount=excluded.last_mount, last_seen=excluded.last_seen""",
                (vol["vol_id"], vol["label"], vol["mount"], now_iso()),
            )
            log(f"\n■ {path}  [区分: {root['zone']}]  ボリューム: {vol['label'] or '(名前なし)'} ({vol['vol_id']})")

            root_failed = False

            def on_error(p, ex, _root=path):
                nonlocal root_failed
                if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(_root)):
                    root_failed = True
                log_error(p, ex)

            for full, name, st in walk_media(path, cfg, on_error):
                volpath = volumes.to_volpath(full, vol["mount"])
                key = (vol["vol_id"], volumes.volkey(volpath))
                if key in seen:
                    continue  # 重なったフォルダ指定で、既に処理済み
                seen.add(key)
                stats["seen"] += 1
                zone = _zone_for(root["zone"], volpath, cfg)

                try:
                    row = conn.execute(
                        """SELECT f.id, f.size, f.mtime_ns, f.asset_id, a.hash_key
                           FROM files f LEFT JOIN assets a ON a.id=f.asset_id
                           WHERE f.vol_id=? AND f.volpath_key=?""",
                        (vol["vol_id"], key[1]),
                    ).fetchone()
                    unchanged = bool(
                        row
                        and row["asset_id"] is not None
                        and row["size"] == st.st_size
                        and row["mtime_ns"] == st.st_mtime_ns
                    )
                    if unchanged and not force_rehash and not redate:
                        conn.execute(
                            "UPDATE files SET status='present', last_run=?, zone=? WHERE id=?",
                            (run_id, zone, row["id"]),
                        )
                        stats["unchanged"] += 1
                    else:
                        # redate: 変更のないファイルは、保存済みのハッシュを使い、日付などだけ読み直す
                        known = row["hash_key"] if (unchanged and redate and not force_rehash and row["hash_key"]) else None
                        info = _process_file(full, name, st, volpath, zone, pets, cfg, known_hash=known, labels=labels)
                        a = conn.execute(
                            "SELECT id FROM assets WHERE hash_key=?", (info["hash_key"],)
                        ).fetchone()
                        if a:
                            asset_id = a["id"]
                        else:
                            asset_id = conn.execute(
                                "INSERT INTO assets(hash_key, kind, size, created_run) VALUES(?,?,?,?)",
                                (info["hash_key"], info["kind"], st.st_size, run_id),
                            ).lastrowid
                        vals = (
                            asset_id, info["ext"], info["kind"], st.st_size, st.st_mtime_ns,
                            info["width"], info["height"], info["duration"], info["zone"],
                            run_id, info["date_taken"], info["date_precision"], info["date_source"],
                            info["date_confidence"], info["date_conflict"], info["date_candidates"],
                            info["hints"], info["camera_make"], info["camera_model"],
                        )
                        if row:
                            conn.execute(
                                """UPDATE files SET asset_id=?, ext=?, kind=?, size=?, mtime_ns=?,
                                   width=?, height=?, duration_sec=?, zone=?, status='present',
                                   last_run=?, date_taken=?, date_precision=?, date_source=?,
                                   date_confidence=?, date_conflict=?, date_candidates=?, hint_pets=?,
                                   camera_make=?, camera_model=?
                                   WHERE id=?""",
                                vals + (row["id"],),
                            )
                            file_id = row["id"]
                            stats["redated" if known else "changed"] += 1
                        else:
                            file_id = conn.execute(
                                """INSERT INTO files(asset_id, ext, kind, size, mtime_ns, width, height,
                                   duration_sec, zone, status, first_run, last_run, date_taken,
                                   date_precision, date_source, date_confidence, date_conflict,
                                   date_candidates, hint_pets, camera_make, camera_model,
                                   vol_id, volpath, volpath_key, filename)
                                   VALUES(?,?,?,?,?,?,?,?,?,'present',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (
                                    asset_id, info["ext"], info["kind"], st.st_size, st.st_mtime_ns,
                                    info["width"], info["height"], info["duration"], info["zone"],
                                    run_id, run_id, info["date_taken"], info["date_precision"],
                                    info["date_source"], info["date_confidence"], info["date_conflict"],
                                    info["date_candidates"], info["hints"],
                                    info["camera_make"], info["camera_model"],
                                    vol["vol_id"], volpath, key[1], name,
                                ),
                            ).lastrowid
                            stats["new"] += 1
                        store_hints(conn, file_id, info["pet_hints"])
                        store_labels(conn, file_id, info["labels"])
                except (OSError, ValueError, UnicodeError) as ex:
                    log_error(full, ex)

                pending += 1
                now = time.time()
                if pending >= BATCH_COMMIT or now - last_log > 15:
                    conn.commit()
                    pending = 0
                if now - last_log >= 10:
                    last_log = now
                    log(
                        f"  ... {stats['seen']:,} 件確認 (新規 {stats['new']:,} / 変更 {stats['changed']:,}"
                        f" / 変化なし {stats['unchanged']:,})  現在: {os.path.dirname(full)}"
                    )

            conn.commit()
            # このフォルダで見つからなかった既存ファイルを「欠損」にする（正常に走査できたときだけ）
            if not root_failed:
                prefix = volumes.volkey(volumes.to_volpath(path, vol["mount"]))
                if prefix:
                    cond = "(volpath_key = ? OR substr(volpath_key, 1, ?) = ?)"
                    args = (vol["vol_id"], run_id, prefix, len(prefix) + 1, prefix + "/")
                else:
                    cond = "1=1"
                    args = (vol["vol_id"], run_id)
                n = conn.execute(
                    f"""UPDATE files SET status='missing'
                        WHERE vol_id=? AND status='present' AND COALESCE(last_run,0)!=? AND {cond}""",
                    args,
                ).rowcount
                stats["missing"] += n
                conn.commit()
    except KeyboardInterrupt:
        interrupted = True
        log("\n中断されました。ここまでの結果は保存済みです。もう一度 scan すると続きから再開します。")

    conn.commit()
    finalize(conn, log)
    status = "interrupted" if interrupted else "done"
    conn.execute(
        """UPDATE scan_runs SET finished=?, status=?, files_seen=?, files_new=?, files_changed=?,
           files_unchanged=?, files_missing=?, errors=? WHERE id=?""",
        (now_iso(), status, stats["seen"], stats["new"], stats["changed"], stats["unchanged"],
         stats["missing"], stats["errors"], run_id),
    )
    conn.commit()

    elapsed = time.time() - t0
    log(
        f"\n完了: {stats['seen']:,} 件確認 / 新規 {stats['new']:,} / 変更 {stats['changed']:,} / "
        f"日付など再計算 {stats['redated']:,} / "
        f"変化なし {stats['unchanged']:,} / 見つからず {stats['missing']:,} / エラー {stats['errors']:,}"
        f"  ({elapsed:.0f}秒)"
    )
    if stats["errors"]:
        log("エラーの詳細は DB の scan_errors テーブル、または export で確認できます。")
    stats["interrupted"] = int(interrupted)
    stats["run_id"] = run_id
    return stats  # Counter: 存在しない項目は 0 として読める


# ---------------------------------------------------------------- 集計
def finalize(conn, log=print) -> None:
    """ファイル単位の情報から、内容（アセット）単位の代表ファイル・日付を決める。"""
    rows = conn.execute(
        """SELECT id, asset_id, size, width, height, duration_sec, zone, status, date_taken,
                  date_precision, date_source, date_confidence, date_conflict
           FROM files ORDER BY asset_id"""
    ).fetchall()

    groups = defaultdict(list)
    for r in rows:
        groups[r["asset_id"]].append(r)

    updates = []
    for asset_id, g in groups.items():
        if asset_id is None:
            continue
        present = [r for r in g if r["status"] == "present"] or g
        master = max(
            present, key=lambda r: ((r["width"] or 0) * (r["height"] or 0), r["size"], -r["id"])
        )
        dated = [r for r in g if r["date_taken"]]
        if dated:
            def _rank(r):
                return (dates.CONF_RANK.get(r["date_confidence"], 0), dates.PREC_RANK.get(r["date_precision"], 0))

            top = max(_rank(r) for r in dated)
            # 同じ確からしさなら、最も早い日付を採る
            # （フォルダ名の日付は「コピーした日」で、撮影日より後になりやすいため）
            best = min((r for r in dated if _rank(r) == top), key=lambda r: (r["date_taken"], r["id"]))
            conflict = int(any(r["date_conflict"] for r in g))
            if not conflict:
                # 同じ内容のコピーは、EXIFなどの撮影日は同じになるはず。フォルダ名の日付が
                # コピーごとに違うのは普通（別の日にコピーしたため）なので、食い違いとは見なさない
                ds = sorted(
                    dt.date.fromisoformat(r["date_taken"][:10])
                    for r in g
                    if r["date_taken"] and r["date_confidence"] in ("high", "medium")
                    and r["date_source"] != "folder"
                )
                if len(ds) >= 2 and (ds[-1] - ds[0]).days > 2:
                    conflict = 1
            d = (best["date_taken"], best["date_precision"], best["date_source"],
                 best["date_confidence"], conflict)
        else:
            d = (None, None, None, "none", 0)
        updates.append(
            (master["id"], sum(1 for r in g if r["status"] == "present"), master["size"], master["width"], master["height"],
             master["duration_sec"], master["zone"], *d, asset_id)
        )

    conn.executemany(
        """UPDATE assets SET master_file_id=?, n_files=?, size=?, width=?, height=?, duration_sec=?,
           zone=?, date_taken=?, date_precision=?, date_source=?, date_confidence=?, date_conflict=?
           WHERE id=?""",
        updates,
    )
    # ファイルが1つも無い内容は削除する。ただしお気に入り等のユーザー情報があるものは残す
    conn.execute(
        """DELETE FROM assets WHERE id NOT IN (SELECT DISTINCT asset_id FROM files WHERE asset_id IS NOT NULL)
           AND id NOT IN (SELECT asset_id FROM asset_user)"""
    )
    conn.execute(
        "UPDATE assets SET n_files=0 WHERE id NOT IN (SELECT DISTINCT asset_id FROM files WHERE asset_id IS NOT NULL)"
    )
    conn.commit()
