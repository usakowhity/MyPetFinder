"""フォルダ名・ファイル名から、場面のタグと場所を拾う（AIは使わない）。

パスの全階層（ファイル名・フォルダ名）を対象にします。上のフォルダの名前も、中のファイルに当てはまります
（例: 「8.8入浴」フォルダの中の写真は、すべて「入浴」）。
呼称が1文字のものは誤って当たりやすいので使いません。短い日本語の呼称は、別の語の一部には当てません。
"""
from __future__ import annotations

from .pethints import _parts, alias_matches
from .textnorm import folder_label


class LabelIndex:
    def __init__(self, conn):
        self.tags = [
            (r["alias_norm"], r["tag_id"])
            for r in conn.execute(
                "SELECT a.alias_norm, a.tag_id FROM tag_aliases a JOIN tags t ON t.id=a.tag_id WHERE t.active=1"
            )
            if len(r["alias_norm"]) >= 2
        ]
        self.places = [
            (r["alias_norm"], r["place_id"])
            for r in conn.execute(
                "SELECT a.alias_norm, a.place_id FROM place_aliases a JOIN places p ON p.id=a.place_id WHERE p.active=1"
            )
            if len(r["alias_norm"]) >= 2
        ]

        import json

        self.rules = {
            r["label"]: (json.loads(r["tags"] or "[]"), json.loads(r["places"] or "[]"))
            for r in conn.execute("SELECT label, tags, places FROM folder_rules")
        }

    @staticmethod
    def _scan(table, levels):
        found: dict[str, str] = {}
        for text in levels:
            for alias, lid in table:
                if lid not in found and alias_matches(alias, text):
                    found[lid] = alias
        return found

    def match(self, volpath: str) -> dict:
        parts = _parts(volpath)
        levels = [n for _, n, _ in parts]
        found = {"tags": self._scan(self.tags, levels), "places": self._scan(self.places, levels)}
        for orig, _, is_folder in parts:          # フォルダ名の規則（上のフォルダの規則も中のファイルに当てはまる）
            if not is_folder:
                continue
            rule = self.rules.get(folder_label(orig))
            if rule:
                for tid in rule[0]:
                    found["tags"].setdefault(tid, orig)
                for pid in rule[1]:
                    found["places"].setdefault(pid, orig)
        return found


def store_labels(conn, file_id: int, labels: dict) -> None:
    """名前由来のタグ・場所だけを入れ替える。人が付けたもの（origin が name 以外）は残す。"""
    conn.execute("DELETE FROM file_tags WHERE file_id=? AND origin='name'", (file_id,))
    conn.execute("DELETE FROM file_places WHERE file_id=? AND origin='name'", (file_id,))
    for tid, matched in labels["tags"].items():
        conn.execute("INSERT OR IGNORE INTO file_tags(file_id, tag_id, origin, matched) VALUES(?,?,'name',?)",
                     (file_id, tid, matched))
    for pid, matched in labels["places"].items():
        conn.execute("INSERT OR IGNORE INTO file_places(file_id, place_id, origin, matched) VALUES(?,?,'name',?)",
                     (file_id, pid, matched))
