"""コマンドライン入口。  python mypet.py <コマンド>"""
from __future__ import annotations

import argparse
import sys

from . import __version__, db, report, scanner
from .config import ConfigError, load_settings


def _setup_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mypet", description="MyPetDatabase 目録作成 (Step 1)")
    ap.add_argument("--config", help="設定ファイル（既定: config/settings.toml）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="データベースを作成し、ペット・人物台帳を読み込む")
    sub.add_parser("preview", help="件数と容量を数える（ハッシュ計算なし・確認用）")
    p = sub.add_parser("scan", help="目録を作成・更新する")
    p.add_argument("--rehash", action="store_true", help="変更がなくても全ファイルを再計算する")
    p.add_argument("--only", help="パスにこの文字列を含む走査先だけを処理する")
    p.add_argument("--redate", action="store_true", help="ハッシュは計算せず、撮影日などだけを読み直す（高速）")
    sub.add_parser("report", help="目録の集計を表示する")
    sub.add_parser("export", help="重複・日付の要確認などをCSVに出力する")
    sub.add_parser("registry", help="ペット・人物台帳を読み込んで表示する")
    p = sub.add_parser("bench", help="AI分析（CLIP）の速度を測る（setup_ai.bat のあとに実行）")
    p.add_argument("--n", type=int, default=200, help="測定に使う画像の枚数（既定 200）")
    p.add_argument("--int8", action="store_true", help="軽量化（int8）モデルで測る")
    p.add_argument("--model", help="画像エンコーダのONNXファイルのパス（ダウンロードできないとき）")
    p.add_argument("--devices", help="測る機器をカンマ区切りで（既定 CPU,GPU,NPU）")
    p = sub.add_parser("embed", help="実写の画像にCLIP特徴ベクトルを付ける（再開可能）")
    p.add_argument("--limit", type=int, help="この件数だけ処理（試し用）")
    p.add_argument("--retry", action="store_true", help="前回失敗したものもやり直す")
    p.add_argument("--model", help="画像エンコーダのONNXファイルのパス")
    p = sub.add_parser("import-review", help="review_ai_folders.csv の「正しいペット」欄を取り込む")
    p.add_argument("file", help="記入済みのCSV（xlsxで保存し直したものも可）")
    sub.add_parser("identify", help="全画像にAIのペット判定を付けて保存する（Embed のあとに実行）")
    sub.add_parser("ratscore", help="ネズミらしさを全画像に付け、Momoの時代の確認用リストを作る")
    p = sub.add_parser("import-photos", help="写真ごとの確認結果（review_momo_era.csv / errors_*.csv）を取り込む")
    p.add_argument("file", help="記入済みのCSV（xlsxで保存し直したものも可）")
    sub.add_parser("nopet", help="ペットが写っていない写真らしさを学習し、確認用リストを作る")
    p = sub.add_parser("web", help="検索画面をブラウザで開く（自分のPCだけから見られる）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p = sub.add_parser("faces", help="写真から顔を探して、顔ごとの特徴を保存する（人物の判定の準備）")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry", action="store_true")
    p = sub.add_parser("video", help="動画からコマを取り出し、コマごとの特徴・顔を保存して、動画のペットを判定する（動画の検索の準備）")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry", action="store_true")
    p.add_argument("--no-faces", action="store_true")
    p = sub.add_parser("petcrop", help="検証: ペット個体の判別を、写真全体と動物の切り出しで比べる（読むだけ）")
    p.add_argument("--per-pet", type=int, default=250)
    p.add_argument("--no-dino", action="store_true")
    p = sub.add_parser("dino", help="全写真に DINOv2 の特徴（ペット個体の判別用）を付ける。終わると CLIP との精度を比べる")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry", action="store_true")
    sub.add_parser("petcompare", help="ペット個体の判別を、CLIP と DINOv2 で比べる")
    sub.add_parser("zeroshot", help="CLIPのゼロショットで、生きたペットか額入り写真・画面・置物かを判定する")
    sub.add_parser("petcheck", help="特徴ベクトルでペットをどこまで当てられるか検証する")
    sub.add_parser("version", help="バージョン表示")
    return ap


def main(argv=None) -> int:
    _setup_stdio()
    args = build_parser().parse_args(argv)
    if args.cmd == "version":
        print(f"MyPetDatabase {__version__}")
        return 0
    try:
        cfg = load_settings(args.config)
    except (ConfigError, FileNotFoundError) as ex:
        print(f"設定エラー: {ex}")
        return 2

    if args.cmd == "preview":
        scanner.run_preview(cfg)
        return 0

    conn = db.connect(cfg)
    try:
        reg = db.sync_registry(conn, cfg)
        for w in reg["warnings"]:
            print("注意:", w)
        if args.cmd == "init":
            print(f"データベース: {cfg.db_path}")
            print(f"ペット {reg['pets']} 件 / 人物 {reg['persons']} 件 / 場所 {reg['places']} 件を台帳に読み込みました。")
        elif args.cmd == "scan":
            scanner.run_scan(cfg, conn, force_rehash=args.rehash, only=args.only, redate=args.redate)
        elif args.cmd == "report":
            report.print_report(conn)
        elif args.cmd == "export":
            report.export_csv(conn, cfg)
        elif args.cmd == "registry":
            report.print_registry(conn)
        elif args.cmd == "embed":
            from . import embed
            return embed.run(cfg, conn, model=args.model, limit=args.limit, retry=args.retry)
        elif args.cmd == "import-review":
            from . import importreview
            return importreview.run(conn, args.file)
        elif args.cmd == "identify":
            from . import identify
            return identify.run(cfg, conn)
        elif args.cmd == "ratscore":
            from . import presence
            return presence.run(cfg, conn)
        elif args.cmd == "import-photos":
            from . import importphotos
            return importphotos.run(conn, args.file)
        elif args.cmd == "nopet":
            from . import nopet
            return nopet.run(cfg, conn)
        elif args.cmd == "web":
            from . import webui
            return webui.serve(cfg, port=args.port, open_browser=not args.no_browser)
        elif args.cmd == "faces":
            from . import faces
            return faces.run(cfg, conn, limit=args.limit, retry=args.retry)
        elif args.cmd == "dino":
            from . import dino
            return dino.run(cfg, conn, limit=args.limit, retry=args.retry)
        elif args.cmd == "petcompare":
            from . import petcheck
            return petcheck.compare(cfg, conn)
        elif args.cmd == "petcrop":
            from . import petcropbench
            return petcropbench.run(cfg, conn, per_pet=args.per_pet, dino=not args.no_dino)
        elif args.cmd == "video":
            from . import video
            return video.run(cfg, conn, limit=args.limit, retry=args.retry, with_faces=not args.no_faces)
        elif args.cmd == "zeroshot":
            from . import zeroshot
            return zeroshot.run(cfg, conn)
        elif args.cmd == "petcheck":
            from . import petcheck
            return petcheck.run(cfg, conn)
        elif args.cmd == "bench":
            from . import bench
            devs = [d.strip().upper() for d in args.devices.split(",")] if args.devices else None
            return bench.run(cfg, conn, n=args.n, kind="int8" if args.int8 else "fp32",
                             model=args.model, devices=devs)
    finally:
        conn.close()
    return 0
