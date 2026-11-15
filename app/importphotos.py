"""写真1枚ごとの確認結果の取り込み（review_momo_era.csv / errors_*.csv など）。

「パス」の列と、いちばん右の記入欄（見出しが「正しいペット」または「写っているのは」で始まる列）を読む。記入欄の書き方:
  ・ペット名（Kuro など。複数は「,」区切り）  → そのペットが写っている（人が確認）
  ・「両方」                                 → Momo と Usako の両方（Momo の時代の確認用リスト専用）
  ・「どちらもいない」「いない」「なし」「-」    → ペットが写っていない。（注）は備考として残す。「額入り写真（Usakoの遺影）」も、生きたペットは写っていないので同じ扱い
  ・空欄                                     → 何もしない
人が確認したペットは file_pets に origin='user', status='identified' で入る。名前の推定やAIの判定のうち、書かれなかったペットは
status='rejected'（写っていない）にする。再スキャンや Identify をやり直しても、この確認は上書きされない。
"""
from __future__ import annotations

import re
import time

from .importreview import NONE_MARKS, SPLIT, _pet_lookup, dir_key
from .report import _mounts, abs_path
from .reviewio import read_table

NONE_WORDS = NONE_MARKS | {"どちらもいない", "いない", "写っていない", "額入り写真", "遺影"}
# 生きたペットが写っていない写真（写真・画面・置物など）の書き出し。（ ）の中の備考は残す
NONE_PREFIXES = ("額入り", "遺影", "TV", "tv", "テレビ", "画面", "モニター", "スマホ画面", "パソコン画面", "写真の写真", "ポスター",
                 "置物", "縫いぐるみ", "ぬいぐるみ", "人物のみ", "人のみ", "景色", "どちらもいない", "いない")


def parse_answer(text: str):
    """(kind, tokens, note)。kind は 'none' / 'pets' / ''（空欄）。"""
    t = (text or "").strip()
    if not t:
        return "", [], ""
    m = re.match(r"^([^(（]*)[(（](.*?)[)）]?\s*$", t)
    base, note = (m.group(1).strip(), m.group(2).strip()) if m else (t, "")
    if base in NONE_WORDS or base.startswith(NONE_PREFIXES):
        # 「どちらもいない」自体は備考にしない。「額入り写真」「TV画面」など種類を表す言葉は備考に残す
        kind_word = "" if base.startswith(("どちらもいない", "いない")) or base in NONE_MARKS else base
        return "none", [], (kind_word + (" " + note if note else "")).strip()
    return "pets", [x for x in SPLIT.split(base) if x], note


def run(conn, path: str, log=print) -> int:
    try:
        rows = read_table(path)
    except Exception as ex:
        log(f"ファイルを読めませんでした: {ex}")
        return 2
    if not rows or "パス" not in rows[0]:
        log("「パス」の列が見つかりません。review_momo_era.csv や errors_*.csv を使ってください。")
        return 2
    col = next((k for k in rows[0] if k.startswith("正しいペット") or k.startswith("写っているのは")), None)
    if col is None:
        log("記入欄（「正しいペット」または「写っているのは」で始まる列）が見つかりません。")
        return 2
    look = _pet_lookup(conn)
    mounts = _mounts(conn)
    want = {dir_key(r["パス"]): r for r in rows if r[col].strip()}
    by_path = {}
    for f in conn.execute("SELECT id, asset_id, vol_id, volpath FROM files WHERE status='present'"):
        k = dir_key(abs_path(mounts, f["vol_id"], f["volpath"]))
        if k in want:
            by_path[k] = (f["id"], f["asset_id"])
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_pets = n_none = n_rej = n_miss = 0
    for k, r in want.items():
        kind, toks, note = parse_answer(r[col])
        if k not in by_path:
            n_miss += 1
            continue
        fid, aid = by_path[k]
        pets = []
        if kind == "pets":
            for tok in toks:
                if tok == "両方":
                    pets += [p for p in ("Momo", "Usako") if look(p)]
                else:
                    p = look(tok)
                    if not p:
                        log(f"  注意: 登録にないペット名 '{tok}' → 取り込みません: {r['パス']}")
                        pets = None
                        break
                    pets.append(look(tok))
            if pets is None:
                continue
            pets = sorted(set(pets))
        conn.execute("INSERT OR REPLACE INTO photo_reviews(path_key, asset_id, answer, pets, note, created) VALUES(?,?,?,?,?,?)",
                     (k, aid, kind, ",".join(pets), note, stamp))
        for p in pets:
            conn.execute("""INSERT OR REPLACE INTO file_pets(file_id, pet_id, status, strength, origin, reason)
                            VALUES(?,?,'identified',NULL,'user','photo review')""", (fid, p))
        for old in conn.execute("SELECT pet_id FROM file_pets WHERE file_id=? AND status='hint'", (fid,)).fetchall():
            if old["pet_id"] not in pets:
                conn.execute("UPDATE file_pets SET status='rejected', origin='user', reason='photo review' WHERE file_id=? AND pet_id=?",
                             (fid, old["pet_id"]))
                n_rej += 1
        if kind == "none":
            n_none += 1
        else:
            n_pets += 1
    conn.commit()
    log(f"取り込み完了: ペットが写っている {n_pets} 枚 / ペットが写っていない {n_none} 枚 / 「写っていない」にした推定 {n_rej} 件"
        + (f" / 目録に見つからない {n_miss} 枚" if n_miss else ""))
    return 0
