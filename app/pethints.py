"""フォルダ名・ファイル名・撮影時期から、写っている「可能性のある」ペットを推定する。

ここで求めるのは「手掛かり（hint）」です。写っていると確認した結果（identified）ではありません。
確認や画像認識の結果は、同じ file_pets 表に origin='user' / 'ai' で、別に記録します。

strength（手掛かりの強さ）
  strong  : 名前が一致し、撮影時期もそのペットの生存期間内（名前は、近い階層のものを優先）
  weak    : 名前は一致するが、撮影時期が生存期間の外（例: Hana は 2008〜2010 年）
  unknown : 名前は一致するが、撮影日が不明・低確度なので期間では判断できない
"""
from __future__ import annotations

import re

from .textnorm import folder_label, norm

# 日本語には単語の区切りがないので、短い呼称（ぽち・たま など）は前後の文字を見て
# 「ぽちーばー」「たまいろ」のような別の語の一部を除く。
_LETTER = "a-zぁ-ゖァ-ヺー"   # 漢字は区切りとして扱う（「ぽちちゃん散歩」）
_SUFFIX = "(?:ちゃん|くん|君|さん|より)?"   # 「ハナより」（＝ハナの写真）などの集めたフォルダ名も当てる
_FOLLOW_PARTICLE = "のがをはもとでにや"   # 「ぽちと散歩」「ぽちの日記」など
_SHORT_MAX = 2                              # 正規化後この長さ以下の日本語呼称は、前後を確認する
FUTURE_SLACK_YEARS = 1                      # フォルダ日付は後日コピーの日。亡くなった翌年までは範囲内とみなす
END_DAY_SLACK = 3                           # 日単位の「亡くなった日」から、この日数までは生存期間内とみなす（お葬式・別れの写真のため）


def _has_kana(s: str) -> bool:
    return any("\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa" for c in s)


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def alias_matches(alias_norm: str, text_norm: str) -> bool:
    """正規化済みの呼称 alias_norm が、正規化済みの文字列 text_norm に含まれるか。"""
    if not alias_norm:
        return False
    # 長い呼称・英字・漢字だけの呼称（命日、風呂 など）は、そのまま含まれていれば当てる。
    # かなを含む短い呼称（くろ、もも）だけ、別の語の一部に当たらないよう前後を確認する。
    if len(alias_norm) > _SHORT_MAX or _is_ascii(alias_norm) or not _has_kana(alias_norm):
        return alias_norm in text_norm
    pat = (
        rf"(?:(?<![{_LETTER}])|(?<=の))"
        rf"{re.escape(alias_norm)}{_SUFFIX}"
        rf"(?:(?=[{_FOLLOW_PARTICLE}])|(?![{_LETTER}]))"
    )
    return re.search(pat, text_norm) is not None


_MEMORIAL = re.compile("命日|回忌|法要|供養|追悼|墓参|お墓")
_SHOOTING_SOURCES = ("exif", "exif_fixed", "filename")   # 撮影した日そのものを表す手掛かり


def _parts(volpath: str) -> list[tuple[str, str, bool]]:
    """パスを、ファイル名（拡張子なし）→ 近いフォルダ → 遠いフォルダ の順に、(原文, 正規化, フォルダか) で並べる。"""
    parts = [p for p in re.split(r"[\\/]+", volpath) if p and not re.fullmatch(r"[A-Za-z]:", p)]
    if not parts:
        return []
    stem = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
    out = [(stem, norm(stem), False)]
    out += [(p, norm(p), True) for p in reversed(parts[:-1])]
    return out


def _levels(volpath: str) -> list[str]:
    return [n for _, n, _ in _parts(volpath)]


def _plus_days(iso: str, n: int) -> str:
    import datetime as _dt
    return (_dt.date.fromisoformat(iso) + _dt.timedelta(days=n)).isoformat()


class PetIndex:
    """台帳（pets / pet_aliases）を読み込んで、手掛かりの計算に使う。"""

    def __init__(self, conn):
        self.periods: dict[str, tuple] = {}
        self.aliases: list[tuple[str, str]] = []
        self.exact: dict[str, tuple] = {}     # 日単位の生存期間（任意）: (開始日, 最終日) の YYYY-MM-DD
        for p in conn.execute("SELECT id, period_start, period_end, start_date, end_date FROM pets WHERE active=1"):
            sd, ed = p["start_date"], p["end_date"]
            ys = int(sd[:4]) if sd else p["period_start"]
            ye = int(ed[:4]) if ed else p["period_end"]
            self.periods[p["id"]] = (ys, ye)
            self.exact[p["id"]] = (sd, ed)
        for r in conn.execute(
            "SELECT a.alias_norm, a.pet_id FROM pet_aliases a JOIN pets p ON p.id=a.pet_id WHERE p.active=1"
        ):
            if len(r["alias_norm"]) >= 2:
                self.aliases.append((r["alias_norm"], r["pet_id"]))
        # フォルダ名の規則（ユーザーが folder_rules.toml で指定した、フォルダ名の語 → ペット）
        import json

        self.rules: dict[str, list[str]] = {}
        for r in conn.execute("SELECT label, pets FROM folder_rules"):
            pets = [p for p in json.loads(r["pets"] or "[]") if p in self.periods]
            if pets:
                self.rules[r["label"]] = pets

    def _names(self, text_norm: str) -> list[str]:
        return sorted({pid for a, pid in self.aliases if alias_matches(a, text_norm)})

    def names_in(self, volpath: str) -> list[str]:
        t = norm(volpath)
        return self._names(t)

    def strength(self, pet_id: str, date_taken, confidence, source=None, conflict=False, collection=False):
        """名前が一致したペットについて、撮影日と生存期間が合うかを判定する。"""
        if not date_taken or confidence not in ("high", "medium") or conflict:
            return "unknown", "撮影日が不明・低確度・手掛かり同士が食い違うため、期間では判断できない"
        year = int(date_taken[:4])
        start, end = self.periods.get(pet_id, (None, None))
        sd, ed = self.exact.get(pet_id, (None, None))
        day = date_taken[:10] if len(date_taken) >= 10 else None        # 日まで分かる撮影日だけ、日単位で比べる
        if day and sd and day < sd:
            return "weak", f"撮影日{day}が迎えた日（{sd}）より前"
        if day and ed and day > _plus_days(ed, END_DAY_SLACK):
            if source in _SHOOTING_SOURCES and not collection:
                return "weak", f"撮影日{day}が亡くなった日（{ed}）より後"
            return "unknown", "日付がコピー・再保存の日付由来、または「〇〇より」の集めたフォルダのため、亡くなった後かは判断できない"
        if start and year < start:
            # コピーした日などの上限の日付でも、生まれる前なら実際の撮影はさらに前なので誤り
            return "weak", f"撮影年{year}が生存期間（{start}〜{end or '?'}）より前"
        if end and year > end + FUTURE_SLACK_YEARS:
            if source in _SHOOTING_SOURCES and not collection:
                return "weak", f"撮影年{year}が生存期間（{start or '?'}〜{end}）より後"
            # コピー・再保存・更新の日付は撮影日より後になるので、亡くなった後かは判断できない
            return "unknown", "日付がコピー・再保存の日付由来、または「〇〇より」の集めたフォルダのため、亡くなった後かは判断できない"
        return "strong", ""

    def hints(self, volpath: str, zone: str, date_taken, confidence, source="exif", conflict=False) -> list[dict]:
        """実写だけが対象（AI生成画像のフォルダ名 Pochi などを、実在のペットと取り違えないため）。

        名前は、ファイル名 → 近いフォルダ → 遠いフォルダ の順に探し、最初に見つかった階層を優先する。
        （例: Pochi遺影候補\\Tamaより\\x.jpg は Tama。遠いフォルダの Pochi は使わない）
        その階層の名前がすべて期間外（weak）のときだけ、さらに上の階層へ進む。
        「命日」「回忌」のフォルダは、名前の個体が写っているとは限らないので期間外と同じ扱いにする。
        """
        if zone != "real":
            return []
        found: dict[str, dict] = {}
        for orig, text, is_folder in _parts(volpath):
            names = self._names(text)
            # ユーザーが指定したフォルダ名の規則のペットを、この階層の名前に加える（命日のフォルダでも降格しない）
            ruled = self.rules.get(folder_label(orig), []) if is_folder else []
            names = sorted(set(names) | set(ruled))
            if not names:
                continue
            memorial = bool(_MEMORIAL.search(text))
            collection = text.endswith("より")   # 「Tamaより」: 亡くなったペットの昔の写真を後から集めたフォルダ
            level = []
            for pid in names:
                if memorial and pid not in ruled:
                    s, reason = "weak", "命日・回忌などのフォルダ名（名前の個体が写っているとは限らない）"
                else:
                    s, reason = self.strength(pid, date_taken, confidence, source, conflict, collection)
                level.append({"pet_id": pid, "strength": s, "reason": reason})
            for h in level:
                found.setdefault(h["pet_id"], h)
            if any(h["strength"] != "weak" for h in level):
                break
        return sorted(found.values(), key=lambda h: h["pet_id"])


def store_hints(conn, file_id: int, hints: list[dict]) -> None:
    """名前由来の手掛かりだけを入れ替える。人や画像認識が付けた結果（origin が name_hint 以外）は残す。"""
    conn.execute("DELETE FROM file_pets WHERE file_id=? AND origin='name_hint'", (file_id,))
    for h in hints:
        conn.execute(
            """INSERT OR IGNORE INTO file_pets(file_id, pet_id, status, strength, origin, reason)
               VALUES(?,?,'hint',?, 'name_hint', ?)""",
            (file_id, h["pet_id"], h["strength"], h["reason"]),
        )
