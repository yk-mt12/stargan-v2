## データベースから画像を取得し、ローカルに保存するプログラム
import sqlite3
import io
from pprint import pprint
from PIL import Image
from datetime import datetime

# データベースに接続
conn = sqlite3.connect('study_model.sqlite3')
c = conn.cursor()

# 画像のバイナリデータを取得する
c.execute("SELECT id, image, taken_date FROM image_data")

image_data = c.fetchall()
count = 0

for data in image_data:
  date_time = datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S')
  hour = date_time.hour

  # バイナリデータをImageオブジェクトに変換する
  img = Image.open(io.BytesIO(data[1]))

  dir_num = hour

  if dir_num not in[12, 18, 21]:
    continue

  # # 画像を保存する
  img.save('./data/custom/train/'+ str(dir_num)+'/'+'flickr_'+str(hour)+'_'+str(count).zfill(6) + '.png')

  count+=1