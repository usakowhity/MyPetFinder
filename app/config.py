"""設定ファイル（TOML）の読み込み。"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info < (3, 11):  # tomllib は 3.11 から標準
    sys.exit("Python 3.11 以上（推奨 3.12）が必要です。")

import tomllib

from .textnorm import norm

APP_ROOT = Path(__file__).resolve().parent.parent
VALID_ZONES = {"real", "ai", "mixed"}


class ConfigError(Exception):
    pass


@dataclass
class Settings:
    app_root: Path
    config_dir: Path
    db_path: Path
    hash_mode: str
    image_exts: frozenset
    video_exts: frozenset
    exclude_dir_names: frozenset
    ai_keywords: tuple
    scans: list = field(default_factory=list)
    exclude_dir_patterns: tuple = ()

    @property
    def media_exts(self) -> frozenset:
        return self.image_exts | self.video_exts


def _is_pattern(name: str) -> bool:
    return "*" in name or "?" in name


def load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise
    except tomllib.TOMLDecodeError as ex:
        raise ConfigError(f"{path} の書式が正しくありません: {ex}") from ex


def load_settings(config_path: str | Path | None = None, app_root: Path | None = None) -> Settings:
    app_root = Path(app_root) if app_root else APP_ROOT
    path = Path(config_path) if config_path else app_root / "config" / "settings.toml"
    if not path.exists():
        raise ConfigError(
            f"設定ファイルが見つかりません: {path}\n"
            "config/settings.example.toml を settings.toml にコピーして編集してください。"
        )
    data = load_toml(path)
    g = data.get("general", {})

    hash_mode = g.get("hash_mode", "full")
    if hash_mode not in ("full", "quick"):
        raise ConfigError('hash_mode は "full" か "quick" にしてください。')

    db_path = Path(g.get("db_path", "data/mypet.db"))
    if not db_path.is_absolute():
        db_path = app_root / db_path

    scans = []
    for s in data.get("scan", []):
        zone = s.get("zone", "real")
        if zone not in VALID_ZONES:
            raise ConfigError(f'zone は real / ai / mixed のどれかにしてください: {s}')
        if "path" not in s:
            raise ConfigError(f"[[scan]] に path がありません: {s}")
        scans.append({"path": str(s["path"]), "zone": zone})
    if not scans:
        raise ConfigError("settings.toml に [[scan]]（走査するフォルダ）が1つもありません。")

    return Settings(
        app_root=app_root,
        config_dir=path.parent,
        db_path=db_path,
        hash_mode=hash_mode,
        image_exts=frozenset(x.lower() for x in g.get("image_exts", [])),
        video_exts=frozenset(x.lower() for x in g.get("video_exts", [])),
        exclude_dir_names=frozenset(x.lower() for x in g.get("exclude_dir_names", []) if not _is_pattern(x)),
        exclude_dir_patterns=tuple(x.lower() for x in g.get("exclude_dir_names", []) if _is_pattern(x)),
        ai_keywords=tuple(norm(x) for x in g.get("ai_path_keywords", []) if x),
        scans=scans,
    )
