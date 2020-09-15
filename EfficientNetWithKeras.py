get_ipython().system("pip install '/kaggle/input/efficientnet/efficientnet-1.0.0-py3-none-any.whl'")
import numpy as np
from PIL import Image
import random
from keras.utils import to_categorical
from keras.optimizers import SGD,Adam
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import models,Sequential,layers
from keras.layers import Activation, BatchNormalization, Dropout, Flatten, Reshape,GlobalAveragePooling2D,SeparableConv2D,Conv2D,Dense,Add, Input,MaxPooling2D
from keras.models import Model
import keras.backend as K
import efficientnet.keras as efn
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from keras.regularizers import l2
import tensorflow as tf

# Numpy 데이터 불러오기 (이미지, 라벨이 묶여서 저장되어 있음)
arr = np.load('/kaggle/input/newdata/new_data_side.npz','r')

X = arr['arr_0']
y = arr['arr_1']

# X = np.r_[arr['arr_0'][:10000],arr['arr_0'][13000:23000]]
# y = np.r_[arr['arr_1'][:10000],arr['arr_1'][13000:23000]]

# 메모리 줄이기 위해 사용
del arr

# Data Shuffle
s = np.arange(X.shape[0])
np.random.shuffle(s)

X = X[s]
y = y[s]

# Model 생성-----------------------------------------------------------
# 직접 Training과 Transfer Learning 두 가지를 진행
# FC 추가, CNN 추가 등 조건을 바꿔가며 실험
EfficientNetB0 = efn.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
EfficientNetB0.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b0_imagenet_1000_notop.h5')

inputs = Input(shape = (224,224,3))
x = EfficientNetB0(inputs)
residual = Conv2D(2048,(1,1),strides=(2,2),padding='same')(x)
x = SeparableConv2D(2048,(3,3),padding='same',strides=1,depth_multiplier=1,depthwise_regularizer=l2(1e-15),pointwise_regularizer=l2(1e-15))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(2048,(3,3),padding='same',strides=1,depth_multiplier=1,depthwise_regularizer=l2(1e-15),pointwise_regularizer=l2(1e-15))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2),padding = "same")(x)
x = Add()([x,residual])
residual1 = Conv2D(2304,(1,1),strides=(2,2),padding='same')(x)
x = SeparableConv2D(2304,(3,3),padding='same',strides=1,depth_multiplier=1,depthwise_regularizer=l2(1e-15),pointwise_regularizer=l2(1e-15))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(2304,(3,3),padding='same',strides=1,depth_multiplier=1,depthwise_regularizer=l2(1e-15),pointwise_regularizer=l2(1e-15))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2),padding = "same")(x)
x = Add()([x,residual1])
residual2 = Conv2D(2560,(1,1),strides=(2,2),padding='same')(x)
x = SeparableConv2D(2560,(3,3),padding='same',strides=1,depth_multiplier=1,depthwise_regularizer=l2(1e-15),pointwise_regularizer=l2(1e-15))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(2560,(3,3),padding='same',strides=1,depth_multiplier=1,depthwise_regularizer=l2(1e-15),pointwise_regularizer=l2(1e-15))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2),padding = "same")(x)
x = Add()([x,residual2])
x = GlobalAveragePooling2D()(x)
x = Dense(1,activation='sigmoid')(x)

model = Model(inputs,x)
model.summary()

# 가중치를 2번째 층까지 고정시켜준다
# for layer in model.layers[:2]:
#     layer.trainable = False

#---------------------------------------------------------------------------------------------------
# Data augmentation 사용하지 않고 훈련
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=321)

del X
del y

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath = 'Efficient_facial_best.hdf5',verbose=1,save_best_only=True)

# optimizer = SGD(lr=0.0007, momentum=0.9)
optimizer = Adam(lr=0.000007, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=1e-7, amsgrad=False)

model.compile(optimizer, loss='binary_crossentropy', metrics=['acc'])

model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2, callbacks=[checkpointer])

from sklearn.metrics import log_loss

y_pred = model.predict(X_test)

logloss = log_loss(y_test,y_pred)
print(logloss)


# ---------------------------------------------------------------------------------------------
# Data Augumentaion 사용하여 훈련
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=321)

del X
del y

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Validation Set은 Data augumentation 안함
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# flow_from_directory는 폴더를 지정해주지만, flow는 데이터를 직접 입력시킨다
# flow에 데이터를 입력할 때, (이미지, 라벨) 형식으로 사용
train_generator = train_datagen.flow(
    # This is the target directory
    (X_train,y_train),
    # All images will be resized to target height and width.
    batch_size=128,
)

validation_generator = test_datagen.flow(
    (X_val,y_val),
    batch_size=128,
)

# fit_generator를 이용하여 훈련
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    epochs = 1,
    steps_per_epoch = 15,
    validation_data = validation_generator,
    validation_steps = 7
)