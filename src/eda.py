import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.visualize import rf_at_once_1, rf_at_once_2, rf_at_once_3, \
    visualize_rf_grid_1, visualize_rf_grid_2, visualize_rf_grid_3
from src.process import x_index_process


rf_path= 'datasets/rf_data'
rf_list= sorted(os.listdir(rf_path))

i= 0
rf_df_2012= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i])))
rf_df_2013= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+1])))
rf_df_2014= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+2])))
rf_df_2015= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+3])))
rf_df_2016= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+4])))
rf_df_2017= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+5])))
rf_df_2018= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+6])))
rf_df_2019= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+7])))
rf_df_2020= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+8])))
rf_df_2021= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+9])))
rf_df_2022= x_index_process(pd.read_csv(os.path.join(rf_path, rf_list[i+10])))

## visalized rainfall each year
## 대곡교 강수량
rf_at_once_1(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022)
## 진관교 강수량
rf_at_once_2(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022)
## 송정동 강수량
rf_at_once_3(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022)


## 대곡교 강수량
visualize_rf_grid_1(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022)
## 진관교 강수량
visualize_rf_grid_2(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022)
## 송정동 강수량
visualize_rf_grid_3(rf_df_2012, rf_df_2013, rf_df_2014, rf_df_2015, rf_df_2016, \
    rf_df_2017, rf_df_2018, rf_df_2019, rf_df_2020, rf_df_2021, rf_df_2022)


## follow up Baseline
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import interpolate

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, AveragePooling1D, GlobalAveragePooling1D

w_list = sorted(glob("datasets/water_data/*.csv"))
pd.read_csv(w_list[0]).shape

## data preprocessing
train_data = []
train_label = []
num = 0

for i in w_list[:-1]:
    
    tmp = pd.read_csv(i)
    tmp = tmp.replace(" ", np.nan)
    tmp = tmp.interpolate(method = 'values')
    tmp = tmp.fillna(0)
    
    for j in tqdm(range(len(tmp)-432)):
        train_data.append(np.array(tmp.loc[j:j + 431, ["swl", "inf", "sfw", "ecpc",
                                                       "tototf", "tide_level",
                                                       "fw_1018662", "fw_1018680",
                                                       "fw_1018683", "fw_1019630"]]).astype(float))
        
        train_label.append(np.array(tmp.loc[j + 432, ["wl_1018662", "wl_1018680",
                                                      "wl_1018683", "wl_1019630"]]).astype(float))

## data split
train_data = np.array(train_data)
train_label = np.array(train_label)

print(train_data.shape)
print(train_label.shape)

## modeling and model learning
input_shape = (train_data[0].shape[0], train_data[0].shape[1])

model = Sequential()
model.add(GRU(256, input_shape=input_shape))
model.add(Dense(4, activation = 'relu'))

# optimizer = tf.optimizers.RMSprop(0.001)
optimizer = tf.optimizers.Adam(0.001)

model.compile(optimizer=optimizer,loss='mse', metrics=['mae'])

model.fit(train_data, train_label, epochs=10, batch_size=128)


################################################################################################################
## building inference datasets
test_data = []
test_label = []

tmp = pd.read_csv(w_list[-1])
tmp = tmp.replace(" ", np.nan)
# 이전값을 사용
tmp = tmp.fillna(method = 'pad')
tmp = tmp.fillna(0)
    
#tmp.loc[:, ["wl_1018662", "wl_1018680", "wl_1018683", "wl_1019630"]] = tmp.loc[:, ["wl_1018662", "wl_1018680", "wl_1018683", "wl_1019630"]]*100
for j in tqdm(range(4032, len(tmp)-432)):
    test_data.append(np.array(tmp.loc[j:j + 431, ["swl", "inf", "sfw", "ecpc",
                                                    "tototf", "tide_level",
                                                    "fw_1018662", "fw_1018680",
                                                    "fw_1018683", "fw_1019630"]]).astype(float))
        
    test_label.append(np.array(tmp.loc[j + 432, ["wl_1018662", "wl_1018680",
                                                    "wl_1018683", "wl_1019630"]]).astype(float))

test_data = np.array(test_data)
test_label = np.array(test_label)

print(test_data.shape)
print(test_label.shape)

pred = model.predict(test_data)
pred = pd.DataFrame(pred)

sample_submission = pd.read_csv("datasets/sample_submission.csv")

sample_submission["wl_1018662"] = pred[0]
sample_submission["wl_1018680"] = pred[1]
sample_submission["wl_1018683"] = pred[2]
sample_submission["wl_1019630"] = pred[3]

sample_submission.to_csv("baseline_1.csv", index = False)









######################################################################################################################
## prediction
## follow up Baseline
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import interpolate

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, AveragePooling1D, GlobalAveragePooling1D

w_list = sorted(glob("datasets/water_data/*.csv"))

## EDA
## find out outliers and missing values
## for example of data_2012
data_2012= pd.read_csv(w_list[0])

import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib as mpl

font_list= [f.name for f in matplotlib.font_manager.fontManager.ttflist if 'Nanum' in f.name]
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'


plt.plot(pd.to_datetime(data_2012.ymdhm), data_2012.swl)
plt.title('팔당댐 현재수위')
plt.show()
data_2012[data_2012.inf.isna()]




## data preprocessing
train_data = []
train_label = []
num = 0

for i in w_list[:-1]:
    
    tmp = pd.read_csv(i)
    tmp = tmp.replace(" ", np.nan)
    tmp = tmp.interpolate(method = 'values')
    tmp = tmp.fillna(0)
    
    for j in tqdm(range(len(tmp)-432)):
        train_data.append(np.array(tmp.loc[j:j + 431, ["swl", "inf", "sfw", "ecpc",
                                                       "tototf", "tide_level",
                                                       "fw_1018662", "fw_1018680",
                                                       "fw_1018683", "fw_1019630"]]).astype(float))
        
        train_label.append(np.array(tmp.loc[j + 432, ["wl_1018662", "wl_1018680",
                                                      "wl_1018683", "wl_1019630"]]).astype(float))

