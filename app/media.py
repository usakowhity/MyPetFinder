"""画像・動画ファイルのメタデータ読み取りとハッシュ計算（標準ライブラリのみ）。

読み取り専用。ファイルを書き換えることはない。
撮影日時が取れない/信用できない場合は None を返し、呼び出し側が
フォルダ名やファイル更新日時から推定する。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import struct

MIN_YEAR = 1990
QUICK_THRESHOLD = 3 * (1 << 20)
_CHUNK = 1 << 20

JPEG_EXTS = {".jpg", ".jpeg", ".jpe"}
MP4_EXTS = {".mp4", ".mov", ".m4v", ".3gp", ".3g2"}
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_MAC_EPOCH = dt.datetime(1904, 1, 1)


def _sane(d: dt.datetime | None) -> bool:
    return d is not None and MIN_YEAR <= d.year <= dt.date.today().year + 1


def _parse_exif_dt(s: str | None):
    if not s:
        return None
    s = s.strip()[:19]
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            d = dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        return d if _sane(d) else None
    return None


# ---------------------------------------------------------------- JPEG / EXIF
def _parse_exif(t: bytes) -> dict:
    """TIFF形式のEXIFから撮影日時を取り出す。"""
    if len(t) < 8:
        return {}
    if t[:2] == b"II":
        e = "<"
    elif t[:2] == b"MM":
        e = ">"
    else:
        return {}
    if struct.unpack(e + "H", t[2:4])[0] != 42:
        return {}
    ifd0_off = struct.unpack(e + "I", t[4:8])[0]

    def read_ifd(off: int) -> dict:
        entries = {}
        if off < 8 or off + 2 > len(t):
            return entries
        n = struct.unpack(e + "H", t[off : off + 2])[0]
        for i in range(min(n, 512)):
            p = off + 2 + i * 12
            if p + 12 > len(t):
                break
            tag, typ, cnt = struct.unpack(e + "HHI", t[p : p + 8])
            entries[tag] = (typ, cnt, t[p + 8 : p + 12])
        return entries

    def ascii_val(entry):
        typ, cnt, raw = entry
        if typ != 2 or cnt == 0:
            return None
        if cnt <= 4:
            data = raw[:cnt]
        else:
            off = struct.unpack(e + "I", raw)[0]
            data = t[off : off + cnt]
        return data.split(b"\x00")[0].decode("ascii", "ignore").strip()

    ifd0 = read_ifd(ifd0_off)
    res: dict = {}
    # カメラのメーカーと機種（日付のずれがどの機種に集中するかの分析に使う）。無ければ記録しない
    for tag, key in ((0x010F, "camera_make"), (0x0110, "camera_model")):
        if tag in ifd0:
            v = ascii_val(ifd0[tag])
            if v:
                res[key] = v[:80]
    if 0x8769 in ifd0:
        sub = read_ifd(struct.unpack(e + "I", ifd0[0x8769][2])[0])
        for tag in (0x9003, 0x9004):  # DateTimeOriginal, DateTimeDigitized
            if tag in sub:
                d = _parse_exif_dt(ascii_val(sub[tag]))
                if d:
                    return {**res, "exif_dt": d, "exif_kind": "exif"}
    if 0x0132 in ifd0:  # DateTime（編集・再保存の日時のことがあるので別扱い）
        d = _parse_exif_dt(ascii_val(ifd0[0x0132]))
        if d:
            return {**res, "exif_dt": d, "exif_kind": "exif_mod"}
    return res


def read_jpeg(path: str) -> dict:
    info: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return info
            got_exif = False
            while True:
                b = f.read(1)
                if not b:
                    break
                if b != b"\xff":
                    continue
                m = f.read(1)
                while m == b"\xff":
                    m = f.read(1)
                if not m:
                    break
                mk = m[0]
                if mk in (0x00, 0x01) or 0xD0 <= mk <= 0xD8:
                    continue
                if mk in (0xD9, 0xDA):
                    break
                lb = f.read(2)
                if len(lb) < 2:
                    break
                seg = struct.unpack(">H", lb)[0] - 2
                if seg < 0:
                    break
                if mk == 0xE1 and not got_exif:
                    data = f.read(seg)
                    if data[:6] == b"Exif\x00\x00":
                        info.update(_parse_exif(data[6:]))
                        got_exif = True
                elif mk in _SOF:
                    data = f.read(seg)
                    if len(data) >= 5:
                        h, w = struct.unpack(">HH", data[1:5])
                        info["height"], info["width"] = h, w
                else:
                    f.seek(seg, 1)
                if got_exif and "width" in info:
                    break
    except (OSError, struct.error):
        pass
    return info


# ---------------------------------------------------------------- その他の画像
def read_image_header(path: str) -> dict:
    """PNG / GIF / BMP の画像サイズ。"""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
    except OSError:
        return {}
    try:
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return {"width": w, "height": h}
        if head[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", head[6:10])
            return {"width": w, "height": h}
        if head[:2] == b"BM" and len(head) >= 26:
            w, h = struct.unpack("<ii", head[18:26])
            return {"width": abs(w), "height": abs(h)}
    except struct.error:
        pass
    return {}


# ---------------------------------------------------------------- MP4 / MOV
def _mac_dt(seconds: int):
    if seconds <= 0:
        return None
    try:
        d = _MAC_EPOCH + dt.timedelta(seconds=seconds)
    except OverflowError:
        return None
    return d if _sane(d) else None


def _box_header(f, pos: int, limit: int):
    f.seek(pos)
    hdr = f.read(8)
    if len(hdr) < 8:
        return None
    size, typ = struct.unpack(">I4s", hdr)
    hl = 8
    if size == 1:
        ext = f.read(8)
        if len(ext) < 8:
            return None
        size = struct.unpack(">Q", ext)[0]
        hl = 16
    elif size == 0:
        size = limit - pos
    if size < hl:
        return None
    return size, typ, hl


def read_mp4(path: str) -> dict:
    info: dict = {}
    try:
        total = os.path.getsize(path)
        with open(path, "rb") as f:
            pos = 0
            while pos + 8 <= total:
                box = _box_header(f, pos, total)
                if not box:
                    break
                size, typ, hl = box
                if typ == b"moov":
                    end = min(pos + size, total)
                    p = pos + hl
                    while p + 8 <= end:
                        sub = _box_header(f, p, end)
                        if not sub:
                            break
                        ssize, styp, shl = sub
                        if styp == b"mvhd":
                            body = f.read(min(ssize - shl, 40))
                            ver = body[0] if body else 0
                            if ver == 1 and len(body) >= 32:
                                ct, _mt, ts, dur = struct.unpack(">QQIQ", body[4:32])
                            elif ver == 0 and len(body) >= 20:
                                ct, _mt, ts, dur = struct.unpack(">IIII", body[4:20])
                            else:
                                break
                            d = _mac_dt(ct)
                            if d:
                                info["container_dt"] = d
                            if ts:
                                info["duration"] = dur / ts
                            break
                        p += ssize
                    break
                pos += size
    except (OSError, struct.error):
        pass
    return info


# ---------------------------------------------------------------- AVI
def read_avi(path: str) -> dict:
    """AVIのヘッダから画像サイズと再生時間を読む（撮影日は入っていないことが多い）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return {}
    if head[:4] != b"RIFF" or head[8:12] != b"AVI ":
        return {}
    i = head.find(b"avih")
    if i < 0 or i + 8 + 40 > len(head):
        return {}
    try:
        usec, _mb, _pad, _fl, frames, _init, _streams, _buf, w, h = struct.unpack(
            "<IIIIIIIIII", head[i + 8 : i + 48]
        )
    except struct.error:
        return {}
    info: dict = {}
    if 0 < w < 100000 and 0 < h < 100000:
        info["width"], info["height"] = w, h
    if usec and frames:
        info["duration"] = usec * frames / 1e6
    return info


def read_meta(path: str, ext: str) -> dict:
    ext = ext.lower()
    if ext in JPEG_EXTS:
        return read_jpeg(path)
    if ext in MP4_EXTS:
        return read_mp4(path)
    if ext == ".avi":
        return read_avi(path)
    if ext in (".png", ".gif", ".bmp"):
        return read_image_header(path)
    return {}


# ---------------------------------------------------------------- ハッシュ
def file_hash(path: str, mode: str = "full") -> str:
    """内容のハッシュ。 'f:'=全体のSHA-256, 'q:'=先頭・末尾・サイズによる簡易版。"""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    buf = bytearray(_CHUNK)
    mv = memoryview(buf)
    with open(path, "rb", buffering=0) as f:
        if mode == "quick" and size > QUICK_THRESHOLD:
            h.update(str(size).encode())
            n = f.readinto(mv)
            h.update(mv[:n])
            f.seek(-_CHUNK, os.SEEK_END)
            n = f.readinto(mv)
            h.update(mv[:n])
            return "q:" + h.hexdigest()
        while True:
            n = f.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
    return "f:" + h.hexdigest()
