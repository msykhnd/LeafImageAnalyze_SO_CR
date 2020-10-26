# Leaf Area Calculator for CR & SQ Image (Ver1.00)

リーフチャンバーおよびホワイトボード上（円筒内）の葉面積を計算するツール

# Target Images

![SQ_Image](https://github.com/msykhnd/LeafImageAnalyze_SQ_CR/blob/master/testimage/test_SQ.JPG)
![CR_Image](https://github.com/msykhnd/LeafImageAnalyze_SQ_CR/blob/master/testimage/test_CR.JPG)

# 特徴
3x3のリーフチャンバーおよび，ホワイトボード上の円筒領域(葉面上にマーキング済み)を対象としています．
HSVによる色相によって葉部分とそれ以外を区別します．

# 利用方法
* ## 共通部 
1. OpenImageタブから葉面積を求めるイメージファイルを指定
2. Openlogから，結果出力先を選択

* ## SQイメージの場合
3. ProcessingからSQImageを選択
4. チャンバーのガラス面の角4点を指定(時計回りor半時計)
選択した場所に赤丸表示
5. EnterでHSVによる境界調整画面へ移動・バーで調整可能
6．調整が終わったらEnterで最終決定
ノイズキャンセルアルゴリズムが起動
7.（注意）
Escキー入力で結果出力，画像選択画面に戻る

* ## CRイメージの場合
3. ProcessingからCRImageを選択
4. 白いプレート上の4角を指定(時計回りor半時計)
選択した場所に赤丸表示
5. Enterで円筒の設置画面へ
6. 葉上の黒枠線の内側4点を指定(順不同)
7. Enterで円筒フィッティングによる円筒内範囲を表示
HSVバーを用いて葉部分の境界調整
8.（注意）
Escキー入力で結果出力，画像選択画面に戻る

# Requipment


