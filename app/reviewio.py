"""確認用CSVの読み込み。Excelで開いて保存し直すと、拡張子は .csv のまま中身が xlsx になることがあるので、両方読めるようにする。"""
from __future__ import annotations

import csv
import io
import re
import zipfile
import xml.etree.ElementTree as ET

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _read_xlsx(path: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % _NS["m"])))
        sheet = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))[0]
        rows = []
        for r in ET.fromstring(z.read(sheet)).iter("{%s}row" % _NS["m"]):
            cells: dict[int, str] = {}
            for c in r.findall("m:c", _NS):
                t = c.get("t")
                v = c.find("m:v", _NS)
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter("{%s}t" % _NS["m"]))
                else:
                    val = v.text if v is not None and v.text is not None else ""
                cells[_col(c.get("r"))] = val
            if cells:
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
        return rows


def _read_csv(path: str) -> list[list[str]]:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp932"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return list(csv.reader(io.StringIO(text)))


def read_table(path: str) -> list[dict]:
    """先頭行を見出しにした辞書の一覧を返す（xlsx でも csv でも）。"""
    with open(path, "rb") as f:
        head = f.read(4)
    rows = _read_xlsx(path) if head[:2] == b"PK" else _read_csv(path)
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    return [dict(zip(header, r + [""] * (len(header) - len(r)))) for r in rows[1:] if any(x.strip() for x in r)]
