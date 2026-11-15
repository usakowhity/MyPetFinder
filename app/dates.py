"""撮影日の推定。

複数の手掛かり（EXIF・ファイル名・フォルダ名・動画の作成日・ファイル更新日）から
最も確からしい日付を選び、確からしさ（confidence）と食い違い（conflict）を記録する。
後で「ペットの生存期間」から個体を判定するとき、日付の確からしさが重要になる。

優先順位（上ほど信頼する）:
  EXIF撮影日 > ファイル名の日付 > フォルダ名の日付 > EXIFの更新日時
  > 動画コンテナの作成日 > フォルダ名の年月 > フォルダ名の年 > ファイル更新日

フォルダ名の日付の扱い（重要）:
  フォルダ名の日付は「SDカードからPCにコピーした日」のことが多く、実際の撮影日より
  後になる。そこで、EXIF・ファイル名・動画の作成日などの「撮影日の手掛かり」が、
  フォルダ名の日付より最大 COPY_WINDOW_DAYS 日前までなら、食い違いとは見なさない。
  逆に、撮影日がフォルダ名の日付より後（コピーした日より後に撮影）は、あり得ないので
  食い違いとする。

確からしさ:
  high   : EXIF撮影日があり食い違いなし / 独立した2つ以上の手掛かりが一致
  medium : フォルダ名・ファイル名・動画作成日の日付が1つだけ（など）
  low    : 年月・年だけ、またはファイル更新日だけ（コピーで変わることがある）
  none   : 手掛かりなし
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import os
import re
import unicodedata

MIN_YEAR = 1990
COPY_WINDOW_DAYS = 92

_RE_YMD8 = re.compile(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
_RE_YMD_SEP = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_.年 ](\d{1,2})[-_.月 ](\d{1,2})(?!\d)")
_RE_YY_SEP = re.compile(r"(?<!\d)(\d{2})[-_.](\d{1,2})[-_.](\d{1,2})(?!\d)")
_RE_YM = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_.年 ](0?[1-9]|1[0-2])(?!\d)")
_RE_Y = re.compile(r"(?<!\d)(19[9]\d|20[0-3]\d)(?!\d)")
# 「2004-2006」のような年の範囲（フォルダ名）。1つの年としては扱わない
_RE_YEAR_RANGE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-－ー～~〜_]\s*((?:19|20)\d{2})(?!\d)")
# 「4.25」「10.26 …」（月.日）や「06-4桜満開」（年-月）のように、数字2つで始まるフォルダ名
_RE_RANGE_END = re.compile(r"[～~〜]\s*(\d{1,2})(?!\d)")
_RE_NUM2 = re.compile(r"^\s*(\d{1,2})[.\-_月](\d{1,2})(?!\d)")
# 「05-夏休み」のように、2桁の数字（年）だけで始まるフォルダ名
_RE_NUM1 = re.compile(r"^\s*(\d{2})[-_.](?!\d)")

PRIORITY = [
    ("exif", "day"),
    ("exif_fixed", "day"),
    ("filename", "day"),
    ("folder", "day"),
    ("exif_mod", "day"),
    ("container", "day"),
    ("folder", "month"),
    ("filename", "month"),
    ("folder", "year"),
    ("mtime", "day"),
]
CONF_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
PREC_RANK = {"year": 1, "month": 2, "day": 3}


def _max_year() -> int:
    return dt.date.today().year + 1


def valid_date(y: int, m: int, d: int):
    if not (MIN_YEAR <= y <= _max_year()):
        return None
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def _yy_to_year(yy: int) -> int:
    return 2000 + yy if yy < 70 else 1900 + yy


def day_from_text(text: str):
    """文字列から年月日を探す。 2016-01-04 / 20160104 / 16-1-4 / 06-08-20 など。"""
    for m in _RE_YMD8.finditer(text):
        d = valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return d
    for m in _RE_YMD_SEP.finditer(text):
        d = valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return d
    for m in _RE_YY_SEP.finditer(text):
        d = valid_date(_yy_to_year(int(m.group(1))), int(m.group(2)), int(m.group(3)))
        if d:
            return d
    return None


def month_from_text(text: str):
    for m in _RE_YM.finditer(text):
        d = valid_date(int(m.group(1)), int(m.group(2)), 1)
        if d:
            return d
    return None


def year_from_text(text: str):
    if _RE_YEAR_RANGE.search(text):
        return None  # 「2004-2006」は範囲であり、1つの年ではない
    for m in _RE_Y.finditer(text):
        d = valid_date(int(m.group(1)), 1, 1)
        if d:
            return d
    return None


def years_of(text: str) -> set:
    """フォルダ名に含まれる年の集合。範囲（2004-2006）は全部の年、単独の年は1つ。"""
    t = unicodedata.normalize("NFKC", text)
    m = _RE_YEAR_RANGE.search(t)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 <= y2 and y2 - y1 <= 40 and y1 >= MIN_YEAR and y2 <= _max_year():
            return set(range(y1, y2 + 1))
        return set()
    d = year_from_text(t)
    return {d.year} if d else set()


def numeric_folder_with_ancestor_year(folders: list[str]):
    """folders は上位→下位の順。最も下位の「数字で始まるフォルダ」を、上位の年入りフォルダと
    組み合わせて日付にする。戻り値は (日付, 精度) か None。

      Pochi2020 / 4.25            → 2020-04-25（日）           月.日の順と解釈
      ポチ2004-2006 / 06-4桜満開 → 2006-04（月）           上位の年と下2桁が一致 → 年-月
      ポチ2004-2006 / 05-夏休み  → 2005（年）
    """
    for i in range(len(folders) - 1, -1, -1):
        text = unicodedata.normalize("NFKC", folders[i])
        m2 = _RE_NUM2.match(text)
        m1 = None if m2 else _RE_NUM1.match(text)
        if not (m2 or m1):
            continue
        years: set = set()
        for j in range(i - 1, -1, -1):
            years = years_of(folders[j])
            if years:
                break
        if not years:
            return None
        if m2:
            a, b = int(m2.group(1)), int(m2.group(2))
            yy_years = sorted(y for y in years if y % 100 == a)
            if yy_years and 1 <= b <= 12:
                d = valid_date(yy_years[0], b, 1)
                return (d, "month") if d else None
            if len(years) == 1:
                d = valid_date(next(iter(years)), a, b)
                return (d, "day") if d else None
            return None
        a = int(m1.group(1))
        yy_years = sorted(y for y in years if y % 100 == a)
        if yy_years:
            d = valid_date(yy_years[0], 1, 1)
            return (d, "year") if d else None
        return None
    return None


def _range_end(d: dt.date, text: str):
    """「08-1-13～19」の「～19」のように、同じ月の中の日付の範囲の終わり。"""
    m = _RE_RANGE_END.search(unicodedata.normalize("NFKC", text))
    if not m or int(m.group(1)) < d.day:
        return None
    return valid_date(d.year, d.month, int(m.group(1)))


def path_candidates(volpath: str, filename: str) -> list[dict]:
    """ファイル名・フォルダ名から日付の手掛かりを集める。"""
    stem = os.path.splitext(filename)[0]
    folders = [c for c in volpath.split("/")[:-1] if c]
    comps = [("filename", stem)] + [("folder", f) for f in reversed(folders)]  # 近い順
    out: list[dict] = []

    for src in ("filename", "folder"):
        for s, text in comps:
            if s != src:
                continue
            d = day_from_text(text)
            if d:
                cand = {"source": src, "precision": "day", "date": d, "text": text}
                if src == "folder":
                    end = _range_end(d, text)
                    if end:
                        cand["end"] = end
                out.append(cand)
                break

    # 「Pochi2020 / 4.25」「ポチ2004-2006 / 06-4桜満開」のように、年が上位フォルダにある場合
    if not any(c["source"] == "folder" and c["precision"] == "day" for c in out):
        r = numeric_folder_with_ancestor_year(folders)
        if r and r[0]:
            out.append({"source": "folder", "precision": r[1], "date": r[0], "text": "numeric+year-folder"})

    for s, text in comps:
        d = month_from_text(text)
        if d:
            out.append({"source": s, "precision": "month", "date": d, "text": text})
            break

    # 年だけの判定は、連番（IMG_2005 など）を誤認しないようフォルダ名に限る
    for s, text in comps:
        if s != "folder":
            continue
        d = year_from_text(text)
        if d:
            out.append({"source": "folder", "precision": "year", "date": d, "text": text})
            break
    return out


def _iso(d) -> str:
    if isinstance(d, dt.datetime):
        return d.isoformat(timespec="seconds")
    return d.isoformat()


def _as_date(d) -> dt.date:
    return d.date() if isinstance(d, dt.datetime) else d


def _period(c: dict):
    d = _as_date(c["date"])
    prec = c["precision"]
    if prec == "month":
        return d.replace(day=1), d.replace(day=calendar.monthrange(d.year, d.month)[1])
    if prec == "year":
        return d.replace(month=1, day=1), d.replace(month=12, day=31)
    return d, (_as_date(c["end"]) if c.get("end") else d)


def rank(c: dict) -> int:
    try:
        return PRIORITY.index((c["source"], c["precision"]))
    except ValueError:
        return 99


def check_pair(a: dict, b: dict):
    """2つの手掛かりが矛盾していれば理由の文字列を、矛盾しなければ None を返す。"""
    pa, pb = _period(a), _period(b)
    fa, fb = a["source"] == "folder", b["source"] == "folder"
    if fa != fb:
        # フォルダ名の日付は「コピーした日」で、撮影日より後になりやすい
        f, o = (a, b) if fa else (b, a)
        pf, po = _period(f), _period(o)
        lo = pf[0] - dt.timedelta(days=COPY_WINDOW_DAYS)
        hi = pf[1] + dt.timedelta(days=2)
        if po[1] < lo:
            if f["precision"] != "day":
                return None  # 年・月だけのフォルダは、古い写真を集めた「まとめフォルダ」のことがある
            base = f"撮影日がフォルダ日付の{COPY_WINDOW_DAYS}日超前"
        elif po[0] > hi:
            base = "撮影日がフォルダ日付より後"
        else:
            return None
        if po[0].year not in (pf[0].year, pf[1].year):
            base += "・年が違う"
        return base
    tol = dt.timedelta(days=2)
    if pa[0] - tol <= pb[1] and pb[0] - tol <= pa[1]:
        return None
    return "年が違う" if {pa[0].year, pa[1].year}.isdisjoint({pb[0].year, pb[1].year}) else "同年・日付が離れている"


def find_conflicts(cands: list[dict]) -> list:
    """矛盾する手掛かりの組を (優先度の高いほう, 低いほう, 理由) の一覧で返す。"""
    precise = [c for c in cands if c["source"] != "mtime"]
    finest = max((PREC_RANK[c["precision"]] for c in precise if c["source"] == "folder"), default=0)
    precise = [c for c in precise if not (c["source"] == "folder" and PREC_RANK[c["precision"]] < finest)]
    out = []
    for i in range(len(precise)):
        for j in range(i + 1, len(precise)):
            a, b = precise[i], precise[j]
            reason = check_pair(a, b)
            if reason:
                if rank(b) < rank(a):
                    a, b = b, a
                out.append((a, b, reason))
    return out


def _add_months(d, n: int):
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    m += 1
    return d.replace(year=y, month=m, day=min(d.day, calendar.monthrange(y, m)[1]))


# 補正後の日付が、フォルダ名の日付（コピーした日）の何日前まで収まっていれば「時計ずれ」と判断するか。
# 広くしすぎると、数日後のEXIFまで「1か月進んでいた」と誤判定するので、コピーの間隔（数日おき）に合わせて狭くする。
CLOCK_FIX_WINDOW_DAYS = 14

# カメラの時計設定ミスの型。EXIFの日時を何か月ずらすと、フォルダ名の日付と整合するか。
# （実データで確認できた型: 月が1か月進んでいた / 年が1年・10年前だった）
CLOCK_SHIFTS = [
    ("月が1か月進んでいた", -1),
    ("年が1年前だった", 12),
    ("年が10年前だった", 120),
]


def fix_clock_errors(cands: list[dict]) -> list[dict]:
    """EXIFがフォルダ名の日付（日単位）と食い違うとき、カメラの時計設定ミスの型に当てはまれば補正する。

    フォルダ名の日付は「コピーした日」＝撮影日の上限に近い。EXIFを決まった月数だけずらしたとき
    その「コピーした日の前14日〜後2日」に収まるなら、時計のずれと判断して補正した日付を採用する。
    元のEXIFの日付は候補の中に（orig として）残す。EXIFは書き換えない。
    """
    folders = [c for c in cands if c["source"] == "folder" and c["precision"] == "day"]
    if not folders:
        return cands
    out = []
    for c in cands:
        if c["source"] == "exif" and c["precision"] == "day" and any(check_pair(c, f) for f in folders):
            for label, months in CLOCK_SHIFTS:
                try:
                    d2 = _add_months(c["date"], months)
                except ValueError:
                    continue
                ok = False
                for f in folders:
                    pf = _period(f)
                    lo = pf[0] - dt.timedelta(days=CLOCK_FIX_WINDOW_DAYS)
                    hi = pf[1] + dt.timedelta(days=2)
                    if lo <= _as_date(d2) <= hi:
                        ok = True
                if ok:
                    c = {"source": "exif_fixed", "precision": "day", "date": d2,
                         "orig": c["date"], "fix": label}
                    break
        out.append(c)
    return out


def choose(cands: list[dict]) -> dict:
    """候補から日付を選び、確からしさと食い違いを判定する。"""
    cands = fix_clock_errors(cands)
    chosen = None
    for src, prec in PRIORITY:
        for c in cands:
            if c["source"] == src and c["precision"] == prec:
                chosen = c
                break
        if chosen:
            break

    cand_json = json.dumps(
        [
            {"source": c["source"], "precision": c["precision"], "date": _iso(c["date"]),
             **({"end": _iso(c["end"])} if c.get("end") else {}),
             **({"orig": _iso(c["orig"]), "fix": c["fix"]} if c.get("orig") else {})}
            for c in cands
        ],
        ensure_ascii=False,
    )
    if not chosen:
        return {
            "date_taken": None,
            "date_precision": None,
            "date_source": None,
            "date_confidence": "none",
            "date_conflict": 0,
            "date_candidates": cand_json,
        }

    precise = [c for c in cands if c["source"] != "mtime"]
    days = [c for c in precise if c["precision"] == "day"]
    conflict = bool(find_conflicts(precise))

    src, prec = chosen["source"], chosen["precision"]
    if prec != "day" or src == "mtime":
        conf = "low"
    elif src == "exif":
        conf = "high"
    elif src == "exif_fixed":
        conf = "medium"  # 補正した日付なので、確実とは言わない
    else:
        independent = {c["source"] for c in days}
        conf = "high" if (len(independent) >= 2 and not conflict) else "medium"
    if conflict:
        conf = {"high": "medium", "medium": "low"}.get(conf, conf)

    d = chosen["date"]
    if prec == "month":
        d = d.replace(day=1)
    elif prec == "year":
        d = d.replace(month=1, day=1)
    return {
        "date_taken": _iso(d),
        "date_precision": prec,
        "date_source": src,
        "date_confidence": conf,
        "date_conflict": 1 if conflict else 0,
        "date_candidates": cand_json,
    }
