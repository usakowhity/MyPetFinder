"""SQLiteデータベース（目録）の定義とペット・人物台帳の同期。"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from .config import Settings, load_toml
from .textnorm import norm

SCHEMA_VERSION = "2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS volumes(
    vol_id TEXT PRIMARY KEY, label TEXT, last_mount TEXT, last_seen TEXT
);

CREATE TABLE IF NOT EXISTS scan_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started TEXT, finished TEXT, status TEXT, hash_mode TEXT,
    files_seen INTEGER DEFAULT 0, files_new INTEGER DEFAULT 0,
    files_changed INTEGER DEFAULT 0, files_unchanged INTEGER DEFAULT 0,
    files_missing INTEGER DEFAULT 0, errors INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scan_errors(
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, path TEXT, message TEXT
);

-- 内容（アセット）: 同じ内容のファイルは1つにまとめる
CREATE TABLE IF NOT EXISTS assets(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    size INTEGER, width INTEGER, height INTEGER, duration_sec REAL,
    zone TEXT,
    master_file_id INTEGER,
    n_files INTEGER DEFAULT 1,
    date_taken TEXT, date_precision TEXT, date_source TEXT,
    date_confidence TEXT DEFAULT 'none', date_conflict INTEGER DEFAULT 0,
    created_run INTEGER
);

-- ファイル（保存場所）: 場所は「ボリュームID + ボリューム内の相対パス」
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER REFERENCES assets(id),
    vol_id TEXT NOT NULL,
    volpath TEXT NOT NULL,
    volpath_key TEXT NOT NULL,
    filename TEXT NOT NULL,
    ext TEXT,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER,
    width INTEGER, height INTEGER, duration_sec REAL,
    zone TEXT,
    status TEXT NOT NULL DEFAULT 'present',
    first_run INTEGER, last_run INTEGER,
    date_taken TEXT, date_precision TEXT, date_source TEXT,
    date_confidence TEXT DEFAULT 'none', date_conflict INTEGER DEFAULT 0,
    date_candidates TEXT,
    hint_pets TEXT,
    UNIQUE(vol_id, volpath_key)
);
CREATE INDEX IF NOT EXISTS idx_files_asset ON files(asset_id);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);

-- ペット台帳・人物台帳（config/*.toml から同期。呼称は複数持てる）
CREATE TABLE IF NOT EXISTS pets(
    id TEXT PRIMARY KEY, display_name TEXT, species TEXT,
    period_start INTEGER, period_end INTEGER, appearance TEXT, notes TEXT,
    active INTEGER DEFAULT 1,
    start_date TEXT, end_date TEXT
);
CREATE TABLE IF NOT EXISTS pet_aliases(
    pet_id TEXT NOT NULL, alias TEXT NOT NULL, alias_norm TEXT NOT NULL,
    PRIMARY KEY(pet_id, alias_norm)
);
CREATE TABLE IF NOT EXISTS persons(
    id TEXT PRIMARY KEY, display_name TEXT, is_owner INTEGER DEFAULT 0, notes TEXT,
    active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS person_aliases(
    person_id TEXT NOT NULL, alias TEXT NOT NULL, alias_norm TEXT NOT NULL,
    PRIMARY KEY(person_id, alias_norm)
);

-- 場所台帳（config/places.toml から同期）。category: indoor / semi_outdoor / outdoor
CREATE TABLE IF NOT EXISTS places(
    id TEXT PRIMARY KEY, name TEXT, category TEXT, notes TEXT, active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS place_aliases(
    place_id TEXT NOT NULL, alias TEXT NOT NULL, alias_norm TEXT NOT NULL,
    PRIMARY KEY(place_id, alias_norm)
);

-- ファイルとペットの関連（1枚に複数のペットが写ることがあるので多対多）。
--   status  : hint（名前などからの推定）/ identified（写っていると確認済み）/ rejected（写っていないと確認済み）
--   strength: strong / weak / unknown（hint の強さ。期間との整合で決まる）
--   origin  : name_hint（フォルダ名・ファイル名）/ user（人が確認）/ ai（画像認識）
CREATE TABLE IF NOT EXISTS file_pets(
    file_id INTEGER NOT NULL, pet_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'hint', strength TEXT, origin TEXT NOT NULL, reason TEXT,
    PRIMARY KEY(file_id, pet_id)
);
CREATE INDEX IF NOT EXISTS idx_file_pets_pet ON file_pets(pet_id);

-- 場面・出来事のタグ（config/tags.toml から同期）と、フォルダ名・ファイル名から拾ったタグ・場所
CREATE TABLE IF NOT EXISTS tags(
    id TEXT PRIMARY KEY, name TEXT, kind TEXT, notes TEXT, active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS tag_aliases(
    tag_id TEXT NOT NULL, alias TEXT NOT NULL, alias_norm TEXT NOT NULL,
    PRIMARY KEY(tag_id, alias_norm)
);
CREATE TABLE IF NOT EXISTS file_tags(
    file_id INTEGER NOT NULL, tag_id TEXT NOT NULL, origin TEXT NOT NULL, matched TEXT,
    PRIMARY KEY(file_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_file_tags_tag ON file_tags(tag_id);
CREATE TABLE IF NOT EXISTS file_places(
    file_id INTEGER NOT NULL, place_id TEXT NOT NULL, origin TEXT NOT NULL, matched TEXT,
    PRIMARY KEY(file_id, place_id)
);
CREATE INDEX IF NOT EXISTS idx_file_places_place ON file_places(place_id);

-- 人が付けた場所（検索画面で付ける）。positive=0 は「この場所ではない」。再スキャン・再分析で消えない
CREATE TABLE IF NOT EXISTS place_labels(
    asset_id INTEGER NOT NULL, place_id TEXT NOT NULL, positive INTEGER NOT NULL DEFAULT 1, created TEXT,
    PRIMARY KEY(asset_id, place_id)
);
CREATE INDEX IF NOT EXISTS idx_place_labels_place ON place_labels(place_id);
-- 人が付けた場所の写真に見た目（CLIP）が似ている写真（AIの推定）。place_labels を元に、しきい値以上を自動で入れる
CREATE TABLE IF NOT EXISTS place_ai(
    place_id TEXT NOT NULL, asset_id INTEGER NOT NULL, sim REAL,
    PRIMARY KEY(place_id, asset_id)
);
CREATE INDEX IF NOT EXISTS idx_place_ai_asset ON place_ai(asset_id);

-- 人が付けた場面・タグ（検索画面で付ける）と、見た目が似ているとしてAIが推定したもの。構造は place_labels / place_ai と同じ
CREATE TABLE IF NOT EXISTS tag_labels(
    asset_id INTEGER NOT NULL, tag_id TEXT NOT NULL, positive INTEGER NOT NULL DEFAULT 1, created TEXT,
    PRIMARY KEY(asset_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_tag_labels_tag ON tag_labels(tag_id);
CREATE TABLE IF NOT EXISTS tag_ai(
    tag_id TEXT NOT NULL, asset_id INTEGER NOT NULL, sim REAL,
    PRIMARY KEY(tag_id, asset_id)
);
CREATE INDEX IF NOT EXISTS idx_tag_ai_asset ON tag_ai(asset_id);

-- 顔（人物の判定用）。座標は画像の幅・高さに対する割合。vec は L2正規化した顔の特徴（float16）。個人データなので、配布版には含めない
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id INTEGER NOT NULL,
    x REAL, y REAL, w REAL, h REAL, score REAL, vec BLOB, created TEXT
);
CREATE INDEX IF NOT EXISTS idx_faces_asset ON faces(asset_id);
CREATE TABLE IF NOT EXISTS face_scan(
    asset_id INTEGER PRIMARY KEY, n INTEGER, err TEXT, created TEXT
);
-- 人が付けた顔の名前（positive=0 は「この人ではない」）と、似ているとしてAIが推定したもの
CREATE TABLE IF NOT EXISTS face_labels(
    face_id INTEGER NOT NULL, person_id TEXT NOT NULL, positive INTEGER NOT NULL DEFAULT 1, created TEXT,
    PRIMARY KEY(face_id, person_id)
);
CREATE INDEX IF NOT EXISTS idx_face_labels_person ON face_labels(person_id);
-- 動画のコマ（等間隔に取り出した数枚）の CLIP 特徴。動画1本の判定は、コマの判定をまとめて行う
CREATE TABLE IF NOT EXISTS video_frames(
    asset_id INTEGER NOT NULL, idx INTEGER NOT NULL, t_sec REAL, vec BLOB,
    PRIMARY KEY(asset_id, idx)
);
-- 動画のコマの DINOv2 特徴（公開版のペット判定用。写真と同じ特徴なので、写真の見本をそのまま使える）
CREATE TABLE IF NOT EXISTS video_pet_frames(
    asset_id INTEGER NOT NULL, idx INTEGER NOT NULL, vec BLOB,
    PRIMARY KEY(asset_id, idx)
);
CREATE TABLE IF NOT EXISTS video_scan(
    asset_id INTEGER PRIMARY KEY, n_frames INTEGER, n_faces INTEGER, err TEXT, created TEXT
);
-- 検索に使わない顔（人形・他人など）。グループ表示・似た顔の候補・自動付けから外す
CREATE TABLE IF NOT EXISTS face_ignore(face_id INTEGER PRIMARY KEY, created TEXT);
CREATE TABLE IF NOT EXISTS pet_ignore(asset_id INTEGER PRIMARY KEY, created TEXT);
CREATE TABLE IF NOT EXISTS face_ai(
    person_id TEXT NOT NULL, face_id INTEGER NOT NULL, sim REAL,
    PRIMARY KEY(person_id, face_id)
);
CREATE INDEX IF NOT EXISTS idx_face_ai_face ON face_ai(face_id);

-- フォルダ名の規則（config/folder_rules.toml から同期）。フォルダ名の語 → ペット・タグ・場所
CREATE TABLE IF NOT EXISTS folder_rules(
    label TEXT PRIMARY KEY, pets TEXT, tags TEXT, places TEXT
);

-- 画像の特徴ベクトル（CLIP）。アセット単位。vec は L2正規化した float16 の並び。失敗したものは vec=NULL と err。
CREATE TABLE IF NOT EXISTS embeddings(
    asset_id INTEGER NOT NULL, model TEXT NOT NULL, dim INTEGER, vec BLOB, err TEXT,
    created TEXT,
    PRIMARY KEY(asset_id, model)
);

-- ネズミらしさ（0=ウサギらしい 〜 1=ネズミらしい）。ネズミだけ／ウサギだけの時代の写真で学習した線形モデルの出力
CREATE TABLE IF NOT EXISTS asset_presence(
    asset_id INTEGER PRIMARY KEY, rat_score REAL, created TEXT
);

-- 写真1枚ごとの確認結果（review_*.csv の記入欄を取り込んだもの）。answer: 'none'（ペットが写っていない）/ 'pets'（pets に書いたペット）
CREATE TABLE IF NOT EXISTS photo_reviews(
    path_key TEXT PRIMARY KEY, asset_id INTEGER, answer TEXT, pets TEXT, note TEXT, created TEXT
);

-- ペットが写っていない写真らしさ（0〜1）。photo_reviews の「いない」を学習したモデルの出力
CREATE TABLE IF NOT EXISTS asset_nopet(
    asset_id INTEGER PRIMARY KEY, score REAL, created TEXT
);

-- 撮影日に登録ペットが誰も生きていない写真（生きた姿ではない）。名前のタグとは別のフラグ
CREATE TABLE IF NOT EXISTS asset_nonlive(
    asset_id INTEGER PRIMARY KEY, kind TEXT, reason TEXT, nopet REAL, created TEXT
);

-- フォルダ単位の確認結果（review_ai_folders.csv の「正しいペット」欄を取り込んだもの）。answer が '-' なら「ペットを特定しない」
CREATE TABLE IF NOT EXISTS folder_reviews(
    folder_key TEXT PRIMARY KEY, folder TEXT, answer TEXT, created TEXT
);

-- ユーザーが付ける情報（お気に入り・評価・メモ）。解析結果とは別テーブルに分け、
-- 再スキャンや再分析で消えないようにする。
CREATE TABLE IF NOT EXISTS asset_user(
    asset_id INTEGER PRIMARY KEY,
    favorite INTEGER DEFAULT 0,
    rating INTEGER,
    notes TEXT,
    updated TEXT
);
"""


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def connect(cfg: Settings, check_same_thread: bool = True) -> sqlite3.Connection:
    Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.db_path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(files)")}
    for c in ("camera_make", "camera_model"):  # 旧版のDBに列を足す
        if c not in cols:
            conn.execute(f"ALTER TABLE files ADD COLUMN {c} TEXT")
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(pets)")}
    for c in ("start_date", "end_date"):  # 日単位の生存期間（任意）
        if c not in pcols:
            conn.execute(f"ALTER TABLE pets ADD COLUMN {c} TEXT")
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)", (SCHEMA_VERSION,))
    conn.commit()
    return conn


def _aliases(primary: list[str], extra) -> list[str]:
    seen, out = set(), []
    for a in list(primary) + [str(x) for x in (extra or [])]:
        a = (a or "").strip()
        n = norm(a)
        if a and n not in seen:
            seen.add(n)
            out.append(a)
    return out


def _iso_date(v):
    """pets.toml の start_date / end_date（"2013-05-19" など。TOMLの日付型も可）を YYYY-MM-DD の文字列にする。"""
    if v is None or v == "":
        return None
    t = str(v).strip().replace("/", "-").replace(".", "-")[:10]
    parts = t.split("-")
    try:
        return "%04d-%02d-%02d" % tuple(int(x) for x in parts[:3]) if len(parts) == 3 else None
    except ValueError:
        return None


def sync_registry(conn: sqlite3.Connection, cfg: Settings) -> dict:
    """pets.toml / persons.toml を読み、台帳テーブルへ反映する。"""
    result = {"pets": 0, "persons": 0, "places": 0, "warnings": []}

    pets_path = cfg.config_dir / "pets.toml"
    if pets_path.exists():
        conn.execute("UPDATE pets SET active=0")
        for p in load_toml(pets_path).get("pet", []):
            pid = str(p["id"])
            disp = str(p.get("display_name", pid))
            conn.execute(
                """INSERT INTO pets(id, display_name, species, period_start, period_end, appearance, notes, active,
                                    start_date, end_date)
                   VALUES(?,?,?,?,?,?,?,1,?,?)
                   ON CONFLICT(id) DO UPDATE SET display_name=excluded.display_name,
                     species=excluded.species, period_start=excluded.period_start,
                     period_end=excluded.period_end, appearance=excluded.appearance,
                     notes=excluded.notes, active=1,
                     start_date=excluded.start_date, end_date=excluded.end_date""",
                (pid, disp, p.get("species"), p.get("period_start"), p.get("period_end"),
                 p.get("appearance", ""), p.get("notes", ""),
                 _iso_date(p.get("start_date")), _iso_date(p.get("end_date"))),
            )
            conn.execute("DELETE FROM pet_aliases WHERE pet_id=?", (pid,))
            for a in _aliases([pid, disp], p.get("aliases")):
                conn.execute(
                    "INSERT OR IGNORE INTO pet_aliases(pet_id, alias, alias_norm) VALUES(?,?,?)",
                    (pid, a, norm(a)),
                )
            result["pets"] += 1
    else:
        result["warnings"].append("config/pets.toml がありません（ペット台帳は空のままです）。")

    persons_path = cfg.config_dir / "persons.toml"
    if persons_path.exists():
        conn.execute("UPDATE persons SET active=0")
        for p in load_toml(persons_path).get("person", []):
            pid = str(p["id"])
            disp = str(p.get("display_name", pid))
            conn.execute(
                """INSERT INTO persons(id, display_name, is_owner, notes, active)
                   VALUES(?,?,?,?,1)
                   ON CONFLICT(id) DO UPDATE SET display_name=excluded.display_name,
                     is_owner=excluded.is_owner, notes=excluded.notes, active=1""",
                (pid, disp, 1 if p.get("is_owner") else 0, p.get("notes", "")),
            )
            conn.execute("DELETE FROM person_aliases WHERE person_id=?", (pid,))
            for a in _aliases([disp], p.get("aliases")):
                conn.execute(
                    "INSERT OR IGNORE INTO person_aliases(person_id, alias, alias_norm) VALUES(?,?,?)",
                    (pid, a, norm(a)),
                )
            result["persons"] += 1
    else:
        result["warnings"].append("config/persons.toml がありません（人物台帳は空のままです）。")

    places_path = cfg.config_dir / "places.toml"
    if places_path.exists():
        conn.execute("UPDATE places SET active=0")
        for p in load_toml(places_path).get("place", []):
            pid = str(p["id"])
            name = str(p.get("name", pid))
            conn.execute(
                """INSERT INTO places(id, name, category, notes, active) VALUES(?,?,?,?,1)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, category=excluded.category,
                     notes=excluded.notes, active=1""",
                (pid, name, p.get("category", "outdoor"), p.get("notes", "")),
            )
            conn.execute("DELETE FROM place_aliases WHERE place_id=?", (pid,))
            for a in _aliases([pid, name], p.get("aliases")):
                conn.execute(
                    "INSERT OR IGNORE INTO place_aliases(place_id, alias, alias_norm) VALUES(?,?,?)",
                    (pid, a, norm(a)),
                )
            result["places"] += 1
    else:
        result["warnings"].append("config/places.toml がありません（場所台帳は空のままです）。")

    tags_path = cfg.config_dir / "tags.toml"
    result["tags"] = 0
    if tags_path.exists():
        conn.execute("UPDATE tags SET active=0")
        for p in load_toml(tags_path).get("tag", []):
            pid = str(p["id"])
            name = str(p.get("name", pid))
            conn.execute(
                """INSERT INTO tags(id, name, kind, notes, active) VALUES(?,?,?,?,1)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, kind=excluded.kind,
                     notes=excluded.notes, active=1""",
                (pid, name, p.get("kind", "scene"), p.get("notes", "")),
            )
            conn.execute("DELETE FROM tag_aliases WHERE tag_id=?", (pid,))
            for a in _aliases([name], p.get("aliases")):   # id は英字なので呼称には含めない
                conn.execute(
                    "INSERT OR IGNORE INTO tag_aliases(tag_id, alias, alias_norm) VALUES(?,?,?)",
                    (pid, a, norm(a)),
                )
            result["tags"] += 1
    else:
        result["warnings"].append("config/tags.toml がありません（場面タグは付きません）。")

    _sync_folder_rules(conn, cfg, result)
    conn.commit()
    return result


RULE_SKIP_ITEMS = {"-", "AI", ""}


def _sync_folder_rules(conn, cfg: Settings, result: dict) -> None:
    """folder_rules.toml を読み、items をペット・場所・タグに振り分けて folder_rules 表に入れる。"""
    import json

    conn.execute("DELETE FROM folder_rules")
    result["rules"] = 0
    path = cfg.config_dir / "folder_rules.toml"
    if not path.exists():
        return
    pet_ids = {r["id"]: r["id"] for r in conn.execute("SELECT id FROM pets WHERE active=1")}
    places = {}
    for r in conn.execute(
        "SELECT p.id, a.alias_norm FROM places p JOIN place_aliases a ON a.place_id=p.id WHERE p.active=1"
    ):
        places[r["alias_norm"]] = r["id"]
    tags = {}
    for r in conn.execute("SELECT t.id, a.alias_norm FROM tags t JOIN tag_aliases a ON a.tag_id=t.id"):
        tags[r["alias_norm"]] = r["id"]
    for r in conn.execute("SELECT id, name FROM tags"):
        tags.setdefault(norm(r["name"]), r["id"])
    for rule in load_toml(path).get("rule", []):
        label = str(rule.get("label", "")).strip()
        if not label:
            continue
        rp, rt, rl = [], [], []
        for item in (str(x).strip() for x in rule.get("items", [])):
            if item in RULE_SKIP_ITEMS:
                continue
            n = norm(item)
            if item in pet_ids:
                rp.append(item)
            elif n in places:
                rl.append(places[n])
            else:
                if n not in tags:   # 既存のタグにない語は、新しいタグにする（id は語そのもの）
                    conn.execute(
                        """INSERT INTO tags(id, name, kind, active) VALUES(?,?, 'folder_rule', 1)
                           ON CONFLICT(id) DO UPDATE SET active=1""",
                        (item, item),
                    )
                    tags[n] = item
                else:
                    conn.execute("UPDATE tags SET active=1 WHERE id=?", (tags[n],))
                rt.append(tags[n])
        conn.execute(
            "INSERT OR REPLACE INTO folder_rules(label, pets, tags, places) VALUES(?,?,?,?)",
            (label, json.dumps(rp, ensure_ascii=False), json.dumps(sorted(set(rt)), ensure_ascii=False),
             json.dumps(rl, ensure_ascii=False)),
        )
        result["rules"] += 1
