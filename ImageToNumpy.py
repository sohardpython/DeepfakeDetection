import os
from PIL import Image
import cv2, sys, re
import pandas as pd
import numpy as np
import random
import numpy as np
import glob
from sklearn.metrics import log_loss
import tensorflow as tf

# 전처리 데이터를 불러오기 위한 Directory 지정
all_train_dirs = glob.glob('/kaggle/input/all-data/dfdc_train_part/' + 'dfdc_train_part_*')


# 이미지를 CascadeClassifier 함수 활용하여 눈으로 인식하는 갯수를 이용하여
# 1개일 시 옆모습, 1개 이외일 시 정면으로 지정하여 데이터셋을 나눔
eye_cascade = cv2.CascadeClassifier('/kaggle/input/haar-cascades-for-face-detection/haarcascade_eye.xml')

def detection_eyes(a):
    roi_gray = a[0:160,0:160]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    cnt_eyes = 0
    for (ex,ey,ew,eh) in eyes:
        cnt_eyes += 1
    return cnt_eyes


# 위의 함수를 이용하여 이미지를 불러와 Numpy 형식으로 변환 (옆,앞모습 데이터셋도 나눔)
def read_image(path,file_list,label_img):
    # a는 한 동영상에서 생성된 이미지의 번호를 입력한다. (원하는 이미지를 저장하게 됨)
    a = [150,250]
    im = []
    label = []
    for i in file_list:
        path_dir = path +'/' + i
        for j in a:
            try:
                # 이미지를 Numpy 형식으로 변환
                img = Image.open(path_dir + '/' + str(j) +'.png')
                arr_img = np.array(img)
                # 눈의 갯수로 옆, 앞모습을 나눔
                if detection_eyes(arr_img) == 1:
#                 if detection_eyes(arr_img) != 1:
                    im.append(list(arr_img))
                    label_list = np.array(label_img.iloc[:, [2]])
                # Labeling 작업
                    if i + '.mp4' in label_list:
                        label.append(1)  # Deepfake
                    else:
                        label.append(0)
            except:
                pass
    return im, label


# read_image 함수를 이용하여 지정한 이미지를 저장하는 작업
X = []
y = []
for i in range(len(all_train_dirs)):
    path = all_train_dirs[i]
    file_list = os.listdir(path)
    label = pd.read_csv(all_train_dirs[i] + '/metadata.csv',delimiter=',')
    img,label = read_image(path,file_list,label)
    X += img
    y += label

X = np.array(X)
y = np.array(y).reshape(-1,1)


# savez를 이용하여 이미지와 라벨을 한번에 저장
# (데이터를 npz형태로 저장하여 사용함으로써, 데이터 불러오는 시간을 줄임)
np.savez('data_facial.npz',X,y)
# np.savez('data_side.npz',X,y)