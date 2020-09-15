import numpy as np
import pandas as pd
import random
import os
import time
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

package_path = '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)
from efficientnet_pytorch import EfficientNet

# Data 불러오기
arr = np.load('/kaggle/input/datafacial/data_facial.npz')

X = arr['arr_0'][:25000]
y = arr['arr_1'][:25000]

# arr = np.load('/kaggle/input/newdata/new_data_side.npz')

# X = np.r_[X,arr['arr_0'][:18000]]
# y = np.r_[y,arr['arr_1'][:18000]]

del arr

# Data Shuffle
s = np.arange(X.shape[0])
np.random.shuffle(s)

X = X[s]
y = y[s]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
del X
del y

# Data Augmentation 사용
# Numpy 형식을 Tensor 형식으로 변환하기 위해 사용
img_size = 120
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class ImageTransform_train:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class ImageTransform_val:
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


transform_train = ImageTransform_train(img_size, mean, std)
transform_val = ImageTransform_val(img_size, mean, std)

# Log_loss 값 계산 시 사용
train_len = len(X_train)
val_len = len(X_val)
dataset_sizes = {'train': train_len, 'val': val_len}

# Numpy를 transforms 함수를 이용하여 Tensor로 변환하기 위하여,
# Numpy -> Image -> Tensor로 변환

img_list = []

for image in X_train:
    try:
        image = Image.fromarray(image)
        image = transform_train(image)
        img_list.append(image)
    except:
        img_list.append(None)

del X_train

img_list_val = []

for image in X_val:
    try:
        image = Image.fromarray(image)
        image = transform_val(image)
        img_list_val.append(image)
    except:
        img_list_val.append(None)

del X_val

y_train = torch.from_numpy(y_train).float()
y_val = torch.from_numpy(y_val).float()


# DataLoader 함수를 사용하여 Data를 묶어줌
class JoinDataset_train(Dataset):
    def __init__(self):
        self.len = y_train.shape[0]
        self.x_data = img_list
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


trainset = JoinDataset_train()
train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)


class JoinDataset_val(Dataset):
    def __init__(self):
        self.len = y_val.shape[0]
        self.x_data = img_list_val
        self.y_data = y_val

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


valset = JoinDataset_val()
val_loader = DataLoader(dataset=valset, batch_size=64, shuffle=True)

dataloders = {'train': train_loader, 'val': val_loader}

# Trained model의 Weights를 사용하기 위하여 경로 지정
weight_path = 'advprop_efficientnet-b0.pth'
trained_weights_path = os.path.join('../input/efficientnetpytorch/EfficientNet-PyTorch/efficientnet_weights',
                                    weight_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# model 생성
model = EfficientNet.from_name('efficientnet-b0')
model.load_state_dict(torch.load(trained_weights_path, map_location=torch.device(device)))
model._fc = nn.Linear(model._fc.in_features, out_features=1)


# # Layer를 지정하여 해당 위치까지 Weights를 고정시킴 (Transfer Learning 시 사용)
# for i,param in enumerate(model.parameters()):
#     param.requires_grad = False
#     print(i)
#     if i == 50:
#         break


# # FC Layer에 추가적으로 Dense를 더함 (Transfer Learning 시 사용)
# fc = nn.Sequential(
#         nn.Linear(1280, 512),
#         nn.ReLU(),
#         nn.Linear(512, 128),
#         nn.ReLU(),
# #         nn.BatchNorm2d(1280),
# #         nn.Dropout(0.8),
#         nn.Linear(128,1)
# 		)
# model._fc = fc

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # 각 Epoch에 Train, Validation을 나눠서 계산시킴
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in dataloders[phase]:
                # Dataloders에서 Input Data를 가져옴
                inputs, labels = data

                # Cuda 사용을 위해, Data의 device 변경
                inputs = inputs.cuda()
                labels = labels.cuda()

                # 변화도 초기화
                optimizer.zero_grad()

                # 순전파
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                outputs = (outputs > 0.5).float()

                # Train 일 경우만 역전파
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (outputs == labels).float().sum()
                # 메모리 공간을 위해 사용
                del loss
                del outputs
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} | Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return model


optimizer = optim.Adam(model.parameters(), lr=0.00002)
criterion = nn.BCEWithLogitsLoss()
num_epochs = 100
model = model.cuda()

model_ft = train_model(model, criterion, optimizer, num_epochs=90)