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
import multiprocessing as mp
########################################保存先の指定など#########################################################
save_dir_name = "2018_06_10" #画像を保存するフォルダ名
current_dir = os.path.dirname(__file__) #このファイルまでの絶対パス
input_dir = os.path.join(current_dir,save_dir_name) #Twitterから収集した画像が保存されるフォルダ
################################################################################################################
#4つの鍵を入力する
# Consumer key
CK = ""
# Consumer secret
CS = ""
# Access token
ATK = ""
# Access token secret
ATS = ""
################################################################################################################

class TwitterImageDownloader():
    """
    HomeTimeLineからツイートを取得し、画像つきのTweetIDを抽出する.
    """
    def __init__(self):
        self.twitter =Twython(app_key=CK, app_secret=CS, oauth_token=ATK, oauth_token_secret=ATS)
     
    def get_timeline(self):
        num_pages       = 1
        tweet_per_page  = 200 #一度に読み込むツイート数 = num_pages * tweet_per_page

        max_id = ""
        media_id_list = []
        url_list = []
        for i in range(num_pages):
            try:
                print("getting hometimeline :" + str(i+1) + "page")
                tw_result = (self.twitter.get_home_timeline(count=tweet_per_page, max_id=max_id) if max_id else self.twitter.get_home_timeline(count=tweet_per_page))
                time.sleep(5)
            except Exception as e:
                print("timeline get error ", e)
                break
            else:
                for result in tw_result:
                    max_id = result["id"]
                    if 'media' in result["entities"]:
                        media = result["extended_entities"]["media"]
                        for url in media:
                            url_list.append(url["media_url"])
                            media_id_list.append(max_id) #画像つきのツイートのtweet_id

        return url_list, media_id_list
 
    def create_folder(self, save_dir):
        try:
            os.mkdir(save_dir)
        except Exception as e:
            print("cannot make dir", e)
        file_list = os.listdir(save_dir)
        return file_list
 
    def get_file(self, url, file_list, save_dir, tweet_id):
        new_file_name = ""
        file_name = str(tweet_id) + "_" + url[url.rfind("/")+1:] #tweet_ID_URL.jpg. これをリツイート時に利用する.
        url_large = "%s:large"%(url)
        if not file_name in file_list:
            save_path = os.path.join(save_dir, file_name)
            try:
                #print("download", url_large)
                url_req = urllib.request.urlopen(url_large)
            except Exception as e:
                print("url open error", url_large, e)
            else:
                #print("saving", save_path)
                img_read = url_req.read()
                img = open(save_path, 'wb')
                img.write(img_read)
                img.close()
                new_file_name = file_name #新たに保存したファイル.保存されたファイルは除外する.
                time.sleep(1)
        #else:
            #print("file already exists", file_name)

        return new_file_name
 
    def download(self):
        new_file_list = []
        save_dir  = os.path.join(current_dir, save_dir_name)
        file_list = self.create_folder(save_dir)

        url_list, media_id_list = self.get_timeline()
        num_urls = len(url_list)
        
        for j, url in enumerate(url_list):
            new_file_name = self.get_file(url, file_list, save_dir, media_id_list[j])
            new_file_list.append(new_file_name)
            #print(str(j+1) + "/" + str(num_urls) + "pictures")
        
        new_file_list = [x for x in new_file_list if x] #空の要素を削除する.

        return new_file_list

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

    def network(self, x, test=False):
        # Input:x -> 3,256,256
        # Convolution_5 -> 16,256,256
        h = PF.convolution(x, 16, (3,3), (1,1), name='Convolution_5')
        # BatchNormalization_9
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
        # PReLU_8
        h = PF.prelu(h, 1, False, name='PReLU_8')
        # Convolution_6
        h = PF.convolution(h, 16, (3,3), (1,1), name='Convolution_6')
        # BatchNormalization_5
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5')
        # PReLU_7
        h = PF.prelu(h, 1, False, name='PReLU_7')
        # Convolution_4
        h = PF.convolution(h, 16, (3,3), (1,1), name='Convolution_4')
        # BatchNormalization_2
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
        # PReLU_6
        h = PF.prelu(h, 1, False, name='PReLU_6')
        # MaxPooling_2 -> 16,128,128
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution_2 -> 32,128,128
        h = PF.convolution(h, 32, (3,3), (1,1), name='Convolution_2')
        # BatchNormalization_4
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
        # PReLU_5
        h = PF.prelu(h, 1, False, name='PReLU_5')
        # MaxPooling -> 32,64,64
        h = F.max_pooling(h, (2,2), (2,2), False)

        # Convolution_3 -> 64,64,64
        h = PF.convolution(h, 64, (3,3), (1,1), name='Convolution_3')
        # BatchNormalization
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
        # PReLU_4
        h = PF.prelu(h, 1, False, name='PReLU_4')
        # MaxPooling_4 -> 64,32,32
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution_7 -> 128,32,32
        h = PF.convolution(h, 128, (3,3), (1,1), name='Convolution_7')
        # BatchNormalization_7
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7')
        # PReLU_3
        h = PF.prelu(h, 1, False, name='PReLU_3')
        # MaxPooling_3 -> 128,16,16
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution_8 -> 256,16,16
        h = PF.convolution(h, 256, (3,3), (1,1), name='Convolution_8')
        # BatchNormalization_10
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
        # PReLU_2
        h = PF.prelu(h, 1, False, name='PReLU_2')
        # MaxPooling_5 -> 256,8,8
        h = F.max_pooling(h, (2,2), (2,2), False)
        # Convolution -> 512,8,8
        h = PF.convolution(h, 512, (3,3), (1,1), name='Convolution')
        # BatchNormalization_8
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_8')
        # PReLU
        h = PF.prelu(h, 1, False, name='PReLU')

        # AveragePooling -> 512,1,1
        h = F.average_pooling(h, (8,8), (8,8))
        # BatchNormalization_6
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
        # PReLU_9
        h = PF.prelu(h, 1, False, name='PReLU_9')
        # Affine -> 1
        h = PF.affine(h, (1,), name='Affine')
        # BatchNormalization_3
        h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
        # y'
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
        amp = 1.0 #リサイズ倍率
        img_width = 256 #目的サイズ
        img_height = 256 #目的サイズ
        x = 0 #トリミング始点
        y = 0 #トリミング始点

        img = cv2.imread(image_path)
        size_x = img.shape[1]
        size_y = img.shape[0]

        #短辺側を基準にリサイズ
        if size_x > size_y:
            amp = img_height / size_y
            img = cv2.resize(img,(int(size_x*amp),img_height))
            x = (img.shape[1] - img_width)//2
            img = img[0:img_height,x:x+img_width] #中央をトリミング

        elif size_x < size_y:
            amp = img_width / size_x
            img = cv2.resize(img,(img_width,int(size_y*amp)))
            y = (img.shape[0] - img_height)//2
            img = img[y:y+img_height,0:img_width] #中央をトリミング

        else: #size_x = size_y
            img = cv2.resize(img,(img_width,img_height))

        img = img / 255.0 #学習時に正規化したための処理
        img = img.transpose(2,0,1) #openCVは(高さ,幅,色)なので転置する必要あり.

        return img

    def pred(self,image_path):
        img = self.image_preproccess(image_path) #入力の画像
        self.x.d = img.reshape(self.x.shape) #画像を(1,3,256,256)の行列に整形し,x.dに代入する.
        self.y.forward() #推論の実行

        return self.y.d[0]
###################################################################################################################
def sub_process(que):
    """
    タイムラインから画像を取得し、その画像に対して推論を行い、イラストのTweetIDをキューに追加する動作を繰り返す.
    """
    pred = Predict() #ネットワークが形成される.
    tw = TwitterImageDownloader()

    while True:
        start_time = time.time()
        new_file_list = tw.download() #タイムラインから画像がダウンロードされ,新たに保存したファイルのリストが返ってくる.
        print("新しい画像の枚数: " + str(len(new_file_list)))

        for image_name in new_file_list:
            image_path = os.path.join(input_dir,image_name)
            
            try:
                y = pred.pred(image_path) #推論の実行
            except:
                y = 1
                print("エラーが発生しました")

            if y<0.5: #イラスト
                tweet_id = image_name.split("_")[0]
                que.put(tweet_id)

        twitterAPI_limit_time = 65 - (time.time() - start_time) #HomeTimeLineの読み込みは60秒に1回.あと何秒待つ必要があるか.
        if twitterAPI_limit_time > 0:
            time.sleep(twitterAPI_limit_time)

def main():
    api = twitter.Api(consumer_key = CK, consumer_secret = CS, access_token_key = ATK, access_token_secret = ATS)

    que = mp.Queue()
    illust_pred = mp.Process(target = sub_process, args = (que,))
    illust_pred.start() #サブプロセスを開始

    while True:
        print("残りのキュー:" + str(que.qsize()))

        try:
            tweet_id = que.get()
            api.PostRetweet(tweet_id)
            print(str(tweet_id) + ":リツイートしました")
            time.sleep(40) #36秒以下にするとAPI制限にかかる.
        except:
            print("リツイート済み")


if __name__ == '__main__':
    main()
