# MTCNN을 사용하기 위해 facenet_pytorch 설치
get_ipython().system('pip install ./facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl')

import os
import glob
import json
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN

# 동영상의 경로 및 라벨의 경로 지정
TRAIN_DIR = 'D:/'
TMP_DIR = 'D:/DeepFake/'
METADATA_PATH = TRAIN_DIR + 'metadata.json'

# 동영상 당 생성 Frame 개수 및 사이즈 지정
SCALE = 0.25
N_FRAMES = 3


# FaceExtractor Class 생성
class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize

    def __call__(self, filename, save_dir):

        # Video Frame 개수
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frames 지정에 따라 Numpy 구간 설정
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Frames을 순차적으로 불러와 Image로 저장
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # 지정된 Resolution으로 변경
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                # 지정된 path에 이미지 저장
                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)

        v_cap.release()


# MTCNN 함수를 face_detector에 저장하여 사용
face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

# FaceExtractor Class 정의
face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE)

# TRAIN_DIR에 저장되어 있는 동영상 path
all_train_videos = glob.glob(os.path.join(TRAIN_DIR, '*.mp4'))

# 동영상 폴더를 지정하여 해당 동영상 이름에 맞는 폴더를 생성 후, 이미지 저장
with torch.no_grad():
    for path in tqdm(all_train_videos):
        file_name = path.split('/')[-1]

        save_dir = os.path.join(TMP_DIR, file_name.split(".")[0])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        face_extractor(path, save_dir)