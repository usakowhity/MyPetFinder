"""呼称・パスの比較用に文字列を正規化する。

全角/半角、大文字/小文字、カタカナ/ひらがなの違いを吸収する。
"""
from __future__ import annotations

import re
import unicodedata


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold().strip()
    out = []
    for ch in s:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:  # カタカナ -> ひらがな
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def folder_label(component: str) -> str:
    """フォルダ名から、日付・番号を除いた「語」の部分を取り出す。例: 「2017.10.15眼振残る？」→「眼振残る?」"""
    t = unicodedata.normalize("NFKC", component).strip()
    t = re.sub(r"^[\d.\-_～~月年日\s]+", "", t)       # 先頭の日付
    t = re.sub(r"[\d.\-_～~\s]+$", "", t)            # 末尾の数字（Tama2011 → Tama）
    return t.strip("()（）[]　 ")
