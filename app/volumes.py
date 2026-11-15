"""ボリューム（ドライブ）の識別。

ドライブ文字（D:, E: ...）は変わることがあるため、Windowsではボリュームの
シリアル番号で「同じドライブ」を判定する。ファイルの場所は
「ボリュームID + ボリューム内の相対パス」で記録する。
"""
from __future__ import annotations

import os
import sys


def _win_info(path: str):
    import ctypes
    from ctypes import wintypes

    drive = os.path.splitdrive(os.path.abspath(path))[0]
    if not drive:
        return None
    root = drive + "\\"
    name_buf = ctypes.create_unicode_buffer(261)
    fs_buf = ctypes.create_unicode_buffer(261)
    serial = wintypes.DWORD(0)
    maxlen = wintypes.DWORD(0)
    flags = wintypes.DWORD(0)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(root),
        name_buf,
        261,
        ctypes.byref(serial),
        ctypes.byref(maxlen),
        ctypes.byref(flags),
        fs_buf,
        261,
    )
    if not ok:
        return None
    return {"vol_id": f"{serial.value:08X}", "label": name_buf.value, "mount": root}


def volume_of(path: str) -> dict:
    """path を含むボリュームの {vol_id, label, mount} を返す。"""
    path = os.path.abspath(path)
    if sys.platform == "win32":
        try:
            info = _win_info(path)
            if info:
                return info
        except Exception:
            pass
        drive = os.path.splitdrive(path)[0]
        return {"vol_id": "letter:" + drive.upper().rstrip(":"), "label": "", "mount": drive + "\\"}
    st = os.stat(path)
    return {"vol_id": f"dev:{st.st_dev}", "label": "", "mount": "/"}


def to_volpath(full: str, mount: str) -> str:
    """絶対パスを、ボリューム内の相対パス（区切りは /）にする。"""
    rel = os.path.relpath(full, mount)
    if rel == ".":
        return ""
    return rel.replace(os.sep, "/")


def from_volpath(mount: str, volpath: str) -> str:
    if not volpath:
        return mount
    return os.path.join(mount, *volpath.split("/"))


def volkey(volpath: str) -> str:
    """ファイルの同一性判定用キー。Windowsは大文字小文字を区別しない。"""
    return volpath.lower() if sys.platform == "win32" else volpath
