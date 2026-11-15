"""config/nopet_folders.toml: 「生きたペットが写っていない」フォルダと「他のウサギ」フォルダの登録。

[[folder]]
path = 'D:\\Pochi2024\\7.4'   # 配下のサブフォルダも含む
kind = "遺影・置物"          # 種類（集計用の自由な名前）
live = false                 # false: 生きた動物はいない（NoPet の学習に使う） / true: 生きた動物はいるが登録ペットではない（学習から除外）
"""
from __future__ import annotations

from collections import defaultdict

from .config import load_toml
from .importreview import dir_key
from .report import _mounts, abs_path


def load(cfg) -> list[dict]:
    p = cfg.config_dir / "nopet_folders.toml"
    if not p.exists():
        return []
    out = []
    for r in load_toml(p).get("folder", []):
        path = str(r.get("path", "")).strip()
        if path:
            out.append({"key": dir_key(path), "path": path, "kind": str(r.get("kind") or "未分類"), "live": bool(r.get("live", False))})
    return out


def assets_by_kind(conn, folders: list[dict]):
    """({(kind, live): {asset_id}}, 見つからなかったフォルダの path のリスト) を返す。"""
    res = defaultdict(set)
    hit = set()
    if not folders:
        return res, []
    mounts = _mounts(conn)
    for f in conn.execute("SELECT asset_id, vol_id, volpath FROM files WHERE status='present' AND asset_id IS NOT NULL"):
        full = abs_path(mounts, f["vol_id"], f["volpath"])
        d = dir_key(full.rsplit("\\", 1)[0] if "\\" in full else full.rsplit("/", 1)[0])
        for fo in folders:
            if d == fo["key"] or d.startswith(fo["key"] + "/"):
                res[(fo["kind"], fo["live"])].add(f["asset_id"])
                hit.add(fo["key"])
    return res, [fo["path"] for fo in folders if fo["key"] not in hit]
