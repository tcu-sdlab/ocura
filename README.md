# ocura

## Requirements
### for General
- python 2.7+ 
  - character_recognition/
  - create_image/create_image_ETL(3 or 6).py
- python 3 
  - create_image/conversion_ETL3.py

## How to use
### character_recognition
##### 1. dataset_predisporsal.py
1. character_recognitionフォルダ下にdataset_imagesというフォルダを作成
2. dataset_images フォルダ下に、文字ごとにフォルダ分けをして画像を入れる(フォルダ名は数字にする)
3. `python dataset_predisporsal.py`コマンドを実行
4. **character_label.npy**と**character_data.npy**というファイルが生成される(それぞれ文字のラベルと画像のデータがデータが格納されている)

##### 2. learn.py
1. `learn.py`コマンドを実行
2. **character_recog_model.json**と**character_recog_model.h5**という学習モデルが生成される

##### 3. recognition.py / recognition_free.py
`recognition.py`は一文字のみを認識でき、`recognition_free.py`は自由に文字を記述した画像から各文字を認識できる

###### recognition.py
1. `python recognition.py name format`を実行（第一引数は画像ファイル名、第二引数はフォーマット）

###### recognition_free.py
1. `python recognition_free.py name format`を実行
2. `recog_images/pro_img/name(画像ファイル名)`フォルダ下に認識結果と各文字を切り出した画像が出力される
