# 第三者の部品とライセンス（MyPetFinder）

MyPetFinder 本体のコードは、MIT ライセンスです（`LICENSE` を参照）。
以下の部品は、**配布物（zip）には含めません**。利用者のパソコンで、初回の設定・解析のときに、それぞれの配布元から取得します。

| 部品 | 用途 | ライセンス | 配布元・メモ |
|---|---|---|---|
| DINOv2 small（Meta） | ペットの個体判別 | Apache License 2.0 | 元のモデル facebook/dinov2-* は Apache-2.0（Hugging Face のモデルカードと、公式リポジトリの License 欄で確認）。取得に使う onnx-community/dinov2-small は変換版で、ライセンス欄は空（元のモデルの条件が引き継がれる） |
| CLIP ViT-B/32（OpenAI） | 場面・文章検索 | MIT | OpenAI のモデルカードに license: mit。ただしモデルカードは「研究用。配備する用途は対象外。限定した場面での画像検索も、十分な検証なしには非推奨」と記載（ライセンスの条項ではなく、利用上の注意）。取得に使う Xenova/clip-vit-base-patch32 は変換版 |
| onnxruntime（Microsoft） | モデルの実行 | MIT | pip で取得 |
| OpenCV（opencv-python） | 動画のコマの取り出し | Apache-2.0（同梱の FFmpeg は LGPL） | pip で取得。再配布するときは FFmpeg の LGPL の表示が必要 |
| NumPy | 数値計算 | BSD-3-Clause | pip で取得 |
| Pillow | 画像の読み込み | HPND（MIT 系） | pip で取得 |
| Python（embeddable） | 実行環境 | PSF License | python.org から取得 |

## 公開前に決めること
1. **配布方法**: モデルを zip に含めない（初回に取得）なら、モデルの再配布の義務は生じません。含めるなら、Apache-2.0 のライセンス文と著作権表示（NOTICE）を同梱します。
2. **本体のライセンス**: MIT に決定（LICENSE を同梱）。
3. **CLIP の注意書き**: 家庭の写真の検索（ペット名・場面）は、モデルカードの「推奨しない用途」に近いので、アプリの説明に「場面検索は目安」と明記する。
4. **ライセンス表示の画面**: アプリ内（または同梱ファイル）に、この一覧を出す。

（これは、公開されている情報をもとにした整理で、法的な助言ではありません。公開前に、各配布元のページで最新の条件を確認してください。）
