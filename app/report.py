"""目録のレポート表示とCSV出力。"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from . import volumes
from .config import Settings
from .scanner import human_size
from .textnorm import folder_label  # noqa: F401  （テストなどが report.folder_label で使う）

ZONE_LABEL = {"real": "実写", "ai": "AI生成", "unknown": "未判定"}
CONF_LABEL = {"high": "高", "medium": "中", "low": "低", "none": "なし"}


def _mounts(conn) -> dict:
    return {r["vol_id"]: r["last_mount"] for r in conn.execute("SELECT vol_id, last_mount FROM volumes")}


def abs_path(mounts: dict, vol_id: str, volpath: str) -> str:
    mount = mounts.get(vol_id, "?:\\")
    return volumes.from_volpath(mount, volpath)


_SRC = {"exif": "EXIF撮影日", "exif_fixed": "EXIF撮影日（時計ずれ補正）", "exif_mod": "EXIFの更新日時", "filename": "ファイル名",
        "folder": "フォルダ名", "container": "動画の作成日", "mtime": "ファイル更新日"}
_PREC = {"day": "の日付", "month": "の年月", "year": "の年"}


def _label(src: str, prec: str) -> str:
    base = _SRC.get(src, src)
    return base + _PREC.get(prec, "") if src in ("folder", "filename") else base


def _dirname(p: str) -> str:
    i = max(p.rfind("\\"), p.rfind("/"))
    return p[:i] if i >= 0 else ""


def _basename(p: str) -> str:
    i = max(p.rfind("\\"), p.rfind("/"))
    return p[i + 1 :]


def conflict_breakdown(conn):
    """日付が食い違うものを、「どの手掛かり同士が」「どう」食い違うかで集計する。"""
    import datetime as dt
    import json
    from . import dates as D

    cnt: Counter = Counter()
    example: dict = {}
    gaps: dict = {}
    folders: dict = {}
    mounts = _mounts(conn)
    rows = conn.execute(
        """SELECT f.vol_id, f.volpath, f.date_candidates, f.date_conflict AS f_conflict
           FROM assets a JOIN files f ON f.id=a.master_file_id
           WHERE a.date_conflict=1 AND a.zone='real'
             AND a.id IN (SELECT asset_id FROM files WHERE status='present')"""
    ).fetchall()
    for r in rows:
        try:
            cands = json.loads(r["date_candidates"] or "[]")
        except ValueError:
            cands = []
        items = []
        for c in cands:
            try:
                it = {"source": c["source"], "precision": c["precision"],
                      "date": dt.datetime.fromisoformat(c["date"])}
                if c.get("end"):
                    it["end"] = dt.datetime.fromisoformat(c["end"])
                items.append(it)
            except (KeyError, ValueError):
                continue
        found = D.find_conflicts(items) if r["f_conflict"] else []
        if found:
            found.sort(key=lambda t: (D.rank(t[0]), D.rank(t[1])))
            a, b, reason = found[0]
            key = f"{_label(a['source'], a['precision'])} と {_label(b['source'], b['precision'])}（{reason}）"
            pa, pb = D._period(a), D._period(b)
            gap = (pa[0] - pb[1]).days if pa[0] > pb[1] else (-(pb[0] - pa[1]).days if pa[1] < pb[0] else 0)
            gaps.setdefault(key, []).append(gap)
        elif not r["f_conflict"]:
            key = f"同じ内容のコピー同士で、日付が{D.COPY_WINDOW_DAYS}日超離れている"
        else:
            key = "その他"
        cnt[key] += 1
        full = abs_path(mounts, r["vol_id"], r["volpath"])
        example.setdefault(key, full)
        folders.setdefault(key, Counter())[_dirname(full)] += 1
    return cnt, example, gaps, folders


def print_report(conn, log=print) -> None:
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731

    n_files = one("SELECT COUNT(*) FROM files WHERE status='present'")
    if n_files == 0 and one("SELECT COUNT(*) FROM files") == 0:
        log("目録はまだ空です。先に scan を実行してください。")
        return

    last = conn.execute(
        "SELECT * FROM scan_runs WHERE status IN ('done','interrupted') ORDER BY id DESC LIMIT 1"
    ).fetchone()
    log("=== MyPetDatabase 目録レポート ===")
    if last:
        log(f"最終スキャン: {last['finished']}（{last['status']}）")

    n_assets = one("SELECT COUNT(*) FROM assets WHERE n_files>0 AND id IN (SELECT asset_id FROM files WHERE status='present')")
    total = one("SELECT COALESCE(SUM(size),0) FROM files WHERE status='present'")
    uniq = one(
        """SELECT COALESCE(SUM(a.size),0) FROM assets a
           WHERE a.id IN (SELECT asset_id FROM files WHERE status='present')"""
    )
    missing = one("SELECT COUNT(*) FROM files WHERE status='missing'")
    log(f"\nファイル数: {n_files:,}（見つからなくなったもの {missing:,}） / 内容の種類（重複を除く）: {n_assets:,}")
    log(f"総容量: {human_size(total)} / 重複を除いた実質容量: {human_size(uniq)}")

    log("\n[種類別・区分別]（ファイル数）")
    for r in conn.execute(
        "SELECT kind, zone, COUNT(*) c, SUM(size) s FROM files WHERE status='present' GROUP BY kind, zone ORDER BY kind, zone"
    ):
        log(f"  {'画像' if r['kind']=='image' else '動画'} / {ZONE_LABEL.get(r['zone'], r['zone'])}: "
            f"{r['c']:,} 件 ({human_size(r['s'])})")

    dup = conn.execute(
        """SELECT COUNT(*) g, COALESCE(SUM(extra),0) e, COALESCE(SUM(waste),0) w FROM (
             SELECT COUNT(*)-1 extra, (COUNT(*)-1)*MAX(size) waste FROM files
             WHERE status='present' GROUP BY asset_id HAVING COUNT(*)>1)"""
    ).fetchone()
    log("\n[完全に同じ内容の重複]")
    log(f"  重複グループ: {dup['g']:,} / 余分なファイル: {dup['e']:,} 件 / 無駄な容量: {human_size(dup['w'])}")
    log("  ※ 縮小・再圧縮したコピー（内容が少し違うもの）は、次のステップで検出します。")

    log("\n[撮影日の確からしさ]（内容の種類ごと）")
    for r in conn.execute(
        """SELECT date_confidence c, COUNT(*) n FROM assets
           WHERE id IN (SELECT asset_id FROM files WHERE status='present')
           GROUP BY date_confidence ORDER BY CASE date_confidence
             WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END"""
    ):
        log(f"  {CONF_LABEL.get(r['c'], r['c'])}: {r['n']:,}")
    nconf = one(
        "SELECT COUNT(*) FROM assets WHERE date_conflict=1 AND zone='real' AND id IN (SELECT asset_id FROM files WHERE status='present')"
    )
    log(f"  日付の手掛かりが食い違っているもの（実写のみ・要確認）: {nconf:,}")
    if nconf:
        cnt, example, gaps, folders = conflict_breakdown(conn)
        log("  [食い違いの内訳]（件数の多い順・上位12。例は1件目のファイル）")
        log("   「差」は 左の手掛かり − 右の手掛かり の日数（+は左のほうが後）。中央値（最小〜最大）")
        for key, n in cnt.most_common(12):
            log(f"    {n:,} 件  {key}")
            g = gaps.get(key)
            if g:
                import statistics

                log(f"        差: {statistics.median(g):+.0f}日（{min(g):+d}〜{max(g):+d}）")
                top = Counter(g).most_common(4)
                if top[0][1] >= 5:  # 同じ差が何件もあるときは、偶然ではなく規則性がある
                    log("        差の内訳: " + " / ".join(f"{d:+d}日×{n}件" for d, n in top))
            fc = folders.get(key)
            if fc and len(fc) > 1:
                log("        多いフォルダ: " + " / ".join(f"{_basename(p)}×{n}" for p, n in fc.most_common(3)))
            log(f"        例: {example[key]}")

    log("\n[年別・実写のみ]（内容の種類ごと。確からしさが低い日付も含む）")
    for r in conn.execute(
        """SELECT COALESCE(substr(date_taken,1,4),'不明') y, COUNT(*) n FROM assets
           WHERE zone='real' AND id IN (SELECT asset_id FROM files WHERE status='present')
           GROUP BY y ORDER BY y"""
    ):
        log(f"  {r['y']}: {r['n']:,}")
    other = one(
        "SELECT COUNT(*) FROM assets WHERE COALESCE(zone,'')!='real' AND id IN (SELECT asset_id FROM files WHERE status='present')"
    )
    log(f"  （AI生成・未判定は別掲: {other:,}）")

    rows = conn.execute(
        """SELECT fp.pet_id, fp.strength, COUNT(*) n FROM file_pets fp
           JOIN files f ON f.id=fp.file_id
           WHERE f.status='present' AND fp.status='hint' GROUP BY fp.pet_id, fp.strength"""
    ).fetchall()
    if rows:
        tbl: dict = {}
        for r in rows:
            tbl.setdefault(r["pet_id"], {})[r["strength"]] = r["n"]
        log("\n[ペット名の手掛かり]（フォルダ名・ファイル名の名前 × 撮影年が生存期間内か。確認済みの結果ではありません）")
        log("   強い=名前一致かつ生存期間内 / 弱い=名前一致だが期間外 / 不明=撮影日が不確かで判断できない")
        for pid, d in sorted(tbl.items(), key=lambda kv: -sum(kv[1].values())):
            log(f"  {pid}: 強い {d.get('strong', 0):,} / 弱い {d.get('weak', 0):,} / 不明 {d.get('unknown', 0):,}")
        ident = conn.execute(
            "SELECT COUNT(*) FROM file_pets WHERE status='identified'"
        ).fetchone()[0]
        log(f"  個体の確認済み（pet_identified）: {ident:,} 件")

    fixed = conn.execute(
        "SELECT date_candidates FROM files WHERE status='present' AND zone='real' AND date_source='exif_fixed'"
    ).fetchall()
    if fixed:
        import json

        fc: Counter = Counter()
        for r in fixed:
            for c in json.loads(r["date_candidates"] or "[]"):
                if c.get("fix"):
                    fc[c["fix"]] += 1
        log("\n[カメラの時計設定ミスと判断して、撮影日を補正したもの]（元のEXIFの日付は候補に残してあります）")
        for k, n in fc.most_common():
            log(f"  {n:,} 件  {k}")

    for title, sql in (
        ("場面・出来事のタグ（フォルダ名・ファイル名から）",
         """SELECT t.name n, COUNT(*) c FROM file_tags ft JOIN tags t ON t.id=ft.tag_id
            JOIN files f ON f.id=ft.file_id WHERE f.status='present' GROUP BY t.id ORDER BY c DESC"""),
        ("場所（フォルダ名・ファイル名から）",
         """SELECT p.name n, COUNT(*) c FROM file_places fp JOIN places p ON p.id=fp.place_id
            JOIN files f ON f.id=fp.file_id WHERE f.status='present' GROUP BY p.id ORDER BY c DESC"""),
    ):
        rows_ = conn.execute(sql).fetchall()
        if rows_:
            log(f"\n[{title}]（ファイル数）")
            for r in rows_[:20]:
                log(f"  {r['n']}: {r['c']:,}")

    cams = conn.execute(
        """SELECT COALESCE(camera_make,'') mk, COALESCE(camera_model,'') md, COUNT(*) n,
                  SUM(CASE WHEN date_conflict=1 THEN 1 ELSE 0 END) c
           FROM files WHERE status='present' AND zone='real' AND (camera_make IS NOT NULL OR camera_model IS NOT NULL)
           GROUP BY mk, md HAVING c>0 ORDER BY c DESC LIMIT 12"""
    ).fetchall()
    if cams:
        log("\n[カメラ機種別・日付の食い違い]（食い違いが多い順。時計の設定ミスなど、機種ごとの偏りを調べる手掛かり）")
        for r in cams:
            name = f"{r['mk']} {r['md']}".strip()
            log(f"  {r['c']:,} / {r['n']:,} 件 ({r['c'] * 100 / r['n']:.1f}%)  {name}")
    nocam = conn.execute(
        "SELECT COUNT(*) FROM files WHERE status='present' AND zone='real' AND kind='image' AND camera_model IS NULL AND camera_make IS NULL"
    ).fetchone()[0]
    log(f"  （実写の画像のうちカメラ機種が読めないもの: {nocam:,} 件。Redateを実行するまでは全件がここに入ります）")

    log("\n[ドライブ別]")
    mounts = _mounts(conn)
    for r in conn.execute(
        "SELECT vol_id, COUNT(*) c, SUM(size) s FROM files WHERE status='present' GROUP BY vol_id"
    ):
        log(f"  {mounts.get(r['vol_id'], '?')} ({r['vol_id']}): {r['c']:,} 件 / {human_size(r['s'])}")

    errs = one("SELECT COUNT(*) FROM scan_errors")
    if errs:
        log(f"\n読み取りエラーの記録: {errs:,} 件（export で errors.csv に出力されます）")


def _write_csv(path: Path, header: list[str], rows) -> int:
    n = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # Excelで開ける
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def export_csv(conn, cfg: Settings, log=print) -> Path:
    out = cfg.db_path.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    mounts = _mounts(conn)

    # 重複
    rows = []
    gno = 0
    for a in conn.execute(
        """SELECT a.id, a.hash_key, a.master_file_id FROM assets a
           WHERE (SELECT COUNT(*) FROM files f WHERE f.asset_id=a.id AND f.status='present')>1
           ORDER BY a.id"""
    ).fetchall():
        gno += 1
        for f in conn.execute(
            "SELECT * FROM files WHERE asset_id=? AND status='present' ORDER BY id", (a["id"],)
        ):
            rows.append([
                gno, a["hash_key"], f["size"], 1 if f["id"] == a["master_file_id"] else 0,
                abs_path(mounts, f["vol_id"], f["volpath"]), ZONE_LABEL.get(f["zone"], f["zone"]),
                f["date_taken"] or "",
            ])
    n1 = _write_csv(
        out / "duplicates.csv",
        ["グループ", "ハッシュ", "サイズ(byte)", "代表(1)", "パス", "区分", "撮影日(推定)"],
        rows,
    )

    # 日付の要確認
    rows = []
    for a in conn.execute(
        """SELECT a.*, f.vol_id, f.volpath, f.date_candidates, f.camera_make, f.camera_model FROM assets a
           JOIN files f ON f.id=a.master_file_id
           WHERE (a.date_confidence IN ('low','none') OR a.date_conflict=1)
             AND a.id IN (SELECT asset_id FROM files WHERE status='present')
           ORDER BY a.date_conflict DESC, a.id"""
    ):
        rows.append([
            a["id"], abs_path(mounts, a["vol_id"], a["volpath"]), ZONE_LABEL.get(a["zone"], a["zone"]),
            a["date_taken"] or "", a["date_source"] or "", CONF_LABEL.get(a["date_confidence"], ""),
            "食い違いあり" if a["date_conflict"] else "", a["date_candidates"] or "",
            f"{a['camera_make'] or ''} {a['camera_model'] or ''}".strip(),
        ])
    n2 = _write_csv(
        out / "review_dates.csv",
        ["アセットID", "パス", "区分", "撮影日(推定)", "根拠", "確からしさ", "食い違い", "手掛かり一覧", "カメラ機種"],
        rows,
    )

    # 時計ずれと判断して撮影日を補正したもの（抜き取り確認用）
    import json as _json

    rows = []
    for f in conn.execute(
        """SELECT vol_id, volpath, date_taken, date_candidates, camera_model FROM files
           WHERE status='present' AND date_source='exif_fixed' ORDER BY volpath"""
    ):
        orig, fix, fd = "", "", ""
        for c in _json.loads(f["date_candidates"] or "[]"):
            if c.get("orig"):
                orig, fix = c["orig"], c.get("fix", "")
            elif c["source"] == "folder" and c["precision"] == "day":
                fd = c["date"][:10]
        rows.append([abs_path(mounts, f["vol_id"], f["volpath"]), orig, f["date_taken"] or "", fd, fix,
                     f["camera_model"] or ""])
    n6 = _write_csv(
        out / "fixed_dates.csv",
        ["パス", "元のEXIF日時", "採用した撮影日", "フォルダ名の日付", "補正の種類", "カメラ機種"],
        rows,
    )

    # ペット名の手掛かりが「すべて期間外（weak）」のファイル。名前から個体を決められないので、確認の対象にする。
    # 最右列「正しいペット」に記入して返してもらえれば、判定規則の見直しに使える。
    rows = [
        [r["pets"], r["reasons"], abs_path(mounts, r["vol_id"], r["volpath"]), r["date_taken"] or "",
         r["date_source"] or "", "あり" if r["date_conflict"] else "", ""]
        for r in conn.execute(
            """SELECT f.vol_id, f.volpath, f.date_taken, f.date_source, f.date_conflict,
                      GROUP_CONCAT(fp.pet_id, ', ') pets, MIN(fp.reason) reasons
               FROM file_pets fp JOIN files f ON f.id=fp.file_id
               WHERE f.status='present' AND fp.status='hint'
               GROUP BY f.id HAVING SUM(CASE WHEN fp.strength!='weak' THEN 1 ELSE 0 END)=0
               ORDER BY f.volpath"""
        )
    ]
    n5 = _write_csv(
        out / "review_pets.csv",
        ["名前の手掛かり(すべて期間外)", "理由", "パス", "撮影日(推定)", "日付の根拠", "日付の食い違い", "正しいペット"],
        rows,
    )

    # フォルダ名の語（日付などを除いたもの）の一覧。タグ・場所の辞書を実データから増やすために使う
    n7 = _write_folder_labels(conn, mounts, out)

    # 見つからなくなったファイル
    rows = [
        [abs_path(mounts, f["vol_id"], f["volpath"]), f["size"]]
        for f in conn.execute("SELECT * FROM files WHERE status='missing' ORDER BY id")
    ]
    n3 = _write_csv(out / "missing.csv", ["パス（以前の場所）", "サイズ(byte)"], rows)

    # エラー
    n4 = _write_csv(
        out / "errors.csv",
        ["実行ID", "パス", "内容"],
        [[e["run_id"], e["path"], e["message"]] for e in conn.execute("SELECT * FROM scan_errors ORDER BY id")],
    )

    log(f"CSVを出力しました: {out}")
    log(f"  duplicates.csv   重複ファイル {n1:,} 行")
    log(f"  review_dates.csv 日付の要確認 {n2:,} 件")
    log(f"  fixed_dates.csv  時計ずれとして補正した撮影日 {n6:,} 件")
    log(f"  folder_labels.csv フォルダ名の語の一覧 {n7:,} 件（タグ・場所を増やす手掛かり）")
    log(f"  review_pets.csv  ペット名がすべて生存期間外で、個体を決められないもの {n5:,} 件")
    log(f"  missing.csv      見つからなくなったファイル {n3:,} 件")
    log(f"  errors.csv       読み取りエラー {n4:,} 件")
    return out


def _write_folder_labels(conn, mounts, out) -> int:
    from collections import defaultdict

    from .pethints import _levels

    count: dict = defaultdict(lambda: [0, set(), ""])
    for r in conn.execute("SELECT vol_id, volpath FROM files WHERE status='present'"):
        parts = [p for p in __import__("re").split(r"[\\/]+", r["volpath"]) if p]
        for i, comp in enumerate(parts[:-1]):          # ファイル名は除き、フォルダ名だけ
            lab = folder_label(comp)
            if len(lab) < 2:
                continue
            c = count[lab]
            c[0] += 1
            c[1].add("/".join(parts[: i + 1]))
            if not c[2]:
                c[2] = abs_path(mounts, r["vol_id"], "\\".join(parts[: i + 1]))
    rows = [[k, v[0], len(v[1]), v[2], ""] for k, v in sorted(count.items(), key=lambda kv: -kv[1][0])]
    return _write_csv(out / "folder_labels.csv",
                      ["フォルダ名の語", "ファイル数", "フォルダ数", "例(フォルダ)", "付けたいタグ・場所(記入欄)"], rows)


def print_registry(conn, log=print) -> None:
    log("=== ペット台帳 ===")
    for p in conn.execute("SELECT * FROM pets WHERE active=1 ORDER BY period_start, id"):
        al = [r["alias"] for r in conn.execute("SELECT alias FROM pet_aliases WHERE pet_id=? ORDER BY rowid", (p["id"],))]
        span = f"{p['period_start'] or '?'}〜{p['period_end'] or '?'}"
        log(f"  {p['id']} ({p['species']}, {span})  呼称: {' / '.join(al)}")
    log("\n=== 人物台帳 ===")
    for p in conn.execute("SELECT * FROM persons WHERE active=1 ORDER BY id"):
        al = [r["alias"] for r in conn.execute("SELECT alias FROM person_aliases WHERE person_id=? ORDER BY rowid", (p["id"],))]
        owner = " [本人]" if p["is_owner"] else ""
        log(f"  {p['id']}{owner}  呼称: {' / '.join(al)}")
    log("\n=== タグ台帳 ===")
    for t in conn.execute("SELECT * FROM tags WHERE active=1 ORDER BY kind, rowid"):
        al = [x["alias"] for x in conn.execute("SELECT alias FROM tag_aliases WHERE tag_id=? ORDER BY rowid", (t["id"],))]
        log(f"  [{t['kind']}] {t['name']}  呼称: {' / '.join(al)}")
    log("\n=== 場所台帳 ===")
    cat_label = {"indoor": "屋内", "semi_outdoor": "半屋外", "outdoor": "屋外"}
    for r in conn.execute("SELECT * FROM places WHERE active=1 ORDER BY category, rowid"):
        al = [x["alias"] for x in conn.execute("SELECT alias FROM place_aliases WHERE place_id=? ORDER BY rowid", (r["id"],))]
        log(f"  [{cat_label.get(r['category'], r['category'])}] {r['name']}  呼称: {' / '.join(al)}")
