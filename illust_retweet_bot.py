#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anacondaで環境を構築したあと,これらのライブラリもインストールする.
conda install scipy scikit-image ipython
pip install nnabla
pip install python-twitter
pip install twython
pip install opencv-python
"""
import os
import time
import urllib.request, urllib.error, urllib.parse
from twython import Twython
import twitter
import cv2
import sys
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
########################################保存先の指定など#########################################################
current_dir = os.path.dirname(__file__) #このファイルまでの絶対パス
input_dir = os.path.join(current_dir,"2018_06_10") #推論を実行したいファイルを保存しておく
################################################################################################################
#4つの鍵を入力する(両サイドの''は残す)
# Consumer key
CK = ""
# Consumer secret
CS = ""
# Access token
ATK = ""
# Access token secret
ATS = ""

api = twitter.Api(consumer_key = CK,
                  consumer_secret = CS,
                  access_token_key = ATK,
                  access_token_secret = ATS
                  )
##############################################tweetDownload########################################################
NUM_PAGES       = 5   
TWEET_PER_PAGE  = 200 
 
# 画像を保存するファイル名
SCREEN_NAMES = "2018_06_10" #収集を開始した日

class TwitterImageDownloader(object):
    """
    Twitterから画像をダウンロードする.以下のサイトからスクリプトを頂いて,TweetIDも返すように変更.
    https://qiita.com/imenurok/items/78d25e892c6557d24810
    """
    def __init__(self):
        super(TwitterImageDownloader, self).__init__()
        self.twitter =Twython(app_key=CK, app_secret=CS, oauth_token=ATK, oauth_token_secret=ATS)
 
    def read_ids(self):
        ids_all = [line.replace('@', '') for line in SCREEN_NAMES.splitlines() if line]
        ids = sorted(list(set(ids_all)))
        return ids
     
    def get_timeline(self, screen_name):
        max_id = ''
        max_id_list = []
        url_list = []
        for i in range(NUM_PAGES):
            try:
                print('getting timeline : @', screen_name, (i+1), 'page')
                #tw_result = (self.twitter.get_user_timeline(screen_name=screen_name, count=TWEET_PER_PAGE, max_id=max_id) if max_id else self.twitter.get_user_timeline(screen_name=screen_name, count=TWEET_PER_PAGE))
                tw_result = (self.twitter.get_home_timeline(count=TWEET_PER_PAGE, max_id=max_id) if max_id else self.twitter.get_home_timeline(count=TWEET_PER_PAGE))
                time.sleep(5)
            except Exception as e:
                print("timeline get error ", e)
                break
            else:
                for result in tw_result:
                    max_id = result['id']
                    if 'media' in result['entities']:
                        media = result['extended_entities']['media']
                        for url in media:
                            url_list.append(url['media_url'])
                            max_id_list.append(max_id) #画像つきのツイートのtweet id
            if len(tw_result) < TWEET_PER_PAGE:
                break
        return url_list,max_id_list
 
    def create_folder(self, save_dir):
        try:
            os.mkdir(save_dir)
        except Exception as e:
            print('cannot make dir', e)
        file_list = os.listdir(save_dir)
        return file_list
 
    def get_file(self, url, file_list, save_dir,tweet_id):
        new_file_name = ""
        file_name = url[url.rfind('/')+1:]
        file_name = str(tweet_id) + "_" +file_name #上のファイル名にtweetID_を追加する.これをリツイート時に利用する.
        url_large = '%s:large'%(url)
        if not file_name in file_list:
            new_file_name = file_name #新たに保存したファイル.保存されたファイルは除外する.
            save_path = os.path.join(save_dir, file_name)
            try:
                print("download", url_large)
                url_req = urllib.request.urlopen(url_large)
            except Exception as e:
                print("url open error", url_large, e)
            else:
                print("saving", save_path)
                img_read = url_req.read()
                img = open(save_path, 'wb')
                img.write(img_read)
                img.close()
                time.sleep(1)
        else:
            print("file already exists", file_name)

        return new_file_name
 
    def download(self):
        new_file_list = []
        screen_name_list = self.read_ids()
        num_users = len(screen_name_list)
        for i, screen_name in enumerate(screen_name_list):
            save_dir  = os.path.join(current_dir, screen_name)
            file_list = self.create_folder(save_dir)
 
            url_list,max_id_list = self.get_timeline(screen_name)
            num_urls = len(url_list)
            
            for j, url in enumerate(url_list):
                new_file_name = self.get_file(url, file_list, save_dir,max_id_list[j])
                new_file_list.append(new_file_name)
                print("%d / %d users, %d / %d pictures"%((i+1), num_users, (j+1), num_urls))
        return new_file_list
###################################################推論実行########################################################
class Predict():
    """
    与えられた画像に対して推論を行う.
    戻り値には,その推論の結果を返す.
    """
    def __init__(self):
        #パラメタの初期化
        nn.clear_parameters()

        #入力変数の準備
        self.x = nn.Variable((1,3,256,256)) #(枚数,色,高さ,幅)

        #パラメタの読み込み
        nn.load_parameters(os.path.join(current_dir,"parameters.h5"))

        #推論ネットワークの構築
        self.y = self.network(self.x,test=True) 

    def network(self,x,test=False):
        # Input -> 3,256,256
        # Convolution -> 32,255,255
        with nn.parameter_scope('Convolution'):
            h = PF.convolution(x, 32, (2,2), (0,0))
        # BatchNormalization_5
        with nn.parameter_scope('BatchNormalization_5'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
        # MaxPooling_3 -> 32,127,127
        h = F.max_pooling(h, (2,2), (2,2), True)
        # ReLU
        h = F.relu(h, True)
        # Convolution_4 -> 32,126,126
        with nn.parameter_scope('Convolution_4'):
            h = PF.convolution(h, 32, (2,2), (0,0))
        # BatchNormalization_2
        with nn.parameter_scope('BatchNormalization_2'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
        # MaxPooling_2 -> 32,63,63
        h = F.max_pooling(h, (2,2), (2,2))
        # ReLU_2
        h = F.relu(h, True)

        # Convolution_2 -> 32,62,62
        with nn.parameter_scope('Convolution_2'):
            h = PF.convolution(h, 32, (2,2), (0,0))
        # BatchNormalization_4
        with nn.parameter_scope('BatchNormalization_4'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
        # ReLU_3
        h = F.relu(h, True)
        # MaxPooling -> 32,31,31
        h = F.max_pooling(h, (2,2), (2,2))
        # Convolution_3 -> 32,30,30
        with nn.parameter_scope('Convolution_3'):
            h = PF.convolution(h, 32, (2,2), (0,0))
        # BatchNormalization
        with nn.parameter_scope('BatchNormalization'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
        # ReLU_4
        h = F.relu(h, True)
        # MaxPooling_4 -> 32,15,15
        h = F.max_pooling(h, (2,2), (2,2))

        # Affine_2 -> 256
        with nn.parameter_scope('Affine_2'):
            h = PF.affine(h, (256,))
        # BatchNormalization_6
        with nn.parameter_scope('BatchNormalization_6'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
        # ReLU_5
        h = F.relu(h, True)
        # Affine_3 -> 128
        with nn.parameter_scope('Affine_3'):
            h = PF.affine(h, (128,))
        # BatchNormalization_7
        with nn.parameter_scope('BatchNormalization_7'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
        # ReLU_6
        h = F.relu(h, True)
        # Affine -> 1
        with nn.parameter_scope('Affine'):
            h = PF.affine(h, (1,))
        # BatchNormalization_3
        with nn.parameter_scope('BatchNormalization_3'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
        # Sigmoid
        h = F.sigmoid(h)
        return h

    def image_preproccess(self, image_path):
        """
        画像を読み込んで,短辺側の解像度を256pixelにリサイズして,中央を切り抜きして正方サイズにする.
        cv2とNNCの画像の取扱の仕様により,転置する.
        戻り値はトリミングされた画像(色,高さ,幅).
        """
        size_x = 0 #画像の横解像度
        size_y = 0 #画像の縦解像度
        x_amp = 1.0 #リサイズ倍率
        y_amp = 1.0 #リサイズ倍率
        img_width = 256 #目的サイズ
        img_height = 256 #目的サイズ
        x = 0 #トリミング始点
        y = 0 #トリミング始点

        img = cv2.imread(image_path)
        size_x = img.shape[1]
        size_y = img.shape[0]

        #短辺側を基準にリサイズ
        if size_x > size_y:
            y_amp = img_height / size_y
            x_amp = y_amp
            img = cv2.resize(img,(int(size_x*x_amp),img_height))
            x = (img.shape[1] - img_width)//2
            img = img[0:img_height,x:x+img_width] #中央をトリミング
        else:
            x_amp = img_width / size_x #x_amp<0
            y_amp = x_amp
            img = cv2.resize(img,(img_width,int(size_y*y_amp)))
            y = (img.shape[0] - img_height)//2
            img = img[y:y+img_height,0:img_width] #中央をトリミング

        img = img / 255.0 #学習時に正規化したための処理
        img = img.transpose(2,0,1) #openCVは(高さ,幅,色)なので転置する必要あり.

        return img

    def pred(self,image_path):
        img = self.image_preproccess(image_path) #入力の画像
        self.x.d = img.reshape(self.x.shape) #画像を(1,3,256,256)の行列に整形し,x.dに代入する.
        self.y.forward() #推論の実行

        return self.y.d[0]
###################################################################################################################

def main():
    pred = Predict() #ネットワークが形成される.
    while True:
        tw = TwitterImageDownloader()
        new_file_list = tw.download()

        print("新しい画像の枚数は"+str(len(new_file_list)))

        for image_name in new_file_list:
            image_path = os.path.join(input_dir,image_name)
            print(image_path)

            try: #入力がグレースケールだったり日本語を含んでいたりするとreshapeやcv2.imreadでエラーを返す.5時間に一回くらい起こる.
                y = pred.pred(image_path)
            except:
                y = 1

            if y<0.5:
                print("イラスト"+str(y))
                tweet_id = image_name.split("_")[0]
                try:
                    api.PostRetweet(tweet_id)
                    print("リツイートしました")
                    time.sleep(1*60) #60秒休む
                except:
                    print("リツイート済み")
            else:
                print("その他  "+str(y))

        time.sleep(5*60) # 5分休む

if __name__ == '__main__':
    main()