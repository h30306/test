import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
import warnings
import os
from ipywidgets import interact
import logging
# 避免產生警告等不必要訊息
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')
tensorflow.autograph.set_verbosity(0)
"""
檔案所在之位置
"""
path = '../Data'

# 建構多特徵訓練資料，可以設定過去時間點與預測時間點的長度來建構
def buildTrain(train, target, pastDay=7, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay][target]))
    return np.array(X_train), np.array(Y_train)

# 建構單特徵訓練資料，可以設定過去時間點與預測時間點的長度來建構，只取target欄位作為特徵
def buildTrain_onetarget(train, target, pastDay=7, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay][[target]]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay][target]))
    return np.array(X_train), np.array(Y_train)

# 混合資料
def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

# 建構單特徵LSTM
def buildOneToOneModel(shape):
    model = Sequential()
    model.add(LSTM(128, input_length=shape[1], input_dim=shape[2], return_sequences=False))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)
    model.compile(loss="mae", optimizer=adam)
    return model

# 建構多特徵LSTM

"""
模型建構概念在implement解釋，這邊之後可刪除
"""
def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(128, input_length=shape[1], input_dim=shape[2], return_sequences=False))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)
    model.compile(loss="mae", optimizer=adam)
    return model

# Jupyter notebook上顯示下拉式選單
@interact(target=['New Deaths', 'New Cases', 'ICU Patients'], location=['Austria', 'Australia', 'Belgium', 'Brazil', 'Canada', 'Denmark', 'France', 'Finland', 'Germany', 'Italy', 'Japan', 'Netherlands', 'New Zealand', 'Philippines', 'Poland', 'South Africa', 'South Korea', 'Turkey', 'United States', 'United Kingdom'])
# 定義使用function直接import到主程式當中
def MachineLearningModeling(target, location):
    # 轉換目標內容，顯示的是New Deaths但欄位名稱是new_deaths
    if target == 'New Deaths': target = 'new_deaths'
    elif target == 'New Cases': target = 'new_cases'
    else: target = 'icu_patients'

    ##### Arima #####

    # 取之前存的最佳參數
    df = pd.read_excel(f'{path}/Model/ARIMA_test.xlsx',engine='openpyxl')
    # 之前的測試資料只有部分有測試...所以如果目標的篩選不到就直接設定ｐ,q為5,2
    """
    p,q是什麼可以參考這個
    https://danzhuibing.github.io/ml_arima_basic.html
    """
    try:
        p = int(df[(df['target'] == target) & (df['location'] == location)]['p'])
        q = int(df[(df['target'] == target) & (df['location'] == location)]['q'])
    except:
        p,q = 5, 2

    # 取得之前整理好的Training Data
    df = pd.read_csv(r"{}/Model/{}/{}.csv".format(path, target, location))
    # 刪除row值全部為1的column
    for x in df.columns:
        if (len(df[x].unique()) == 1) & (x != target):
            df.drop(x, 1, inplace=True)

    # Split Training Data to train and test dataset        
    value = df[target].values # 目標變數
    # 0.7作為訓練集
    length = int(len(value) * 0.7) # 訓練集筆數
    train1 = list(value[0:length]) # 訓練集
     # 0.2作為測試集
    test1 = list(value[int(len(value) * 0.8):]) # 測試集

    # Build Model and Training
    print('Training ARIMA Model...')
    # 預設ARIMA model
    model = ARIMA(train1, order=(p, 1, q))
    # 訓練模型
    model_fit = model.fit()
    # 設定儲存預測值的list
    predictions_ARIMA = []
    # 將每個資料丟進去預測以及加入ARIMA模型做
    for i in range(len(test1)):
        # 預測下一個時間點
        pred = model_fit.forecast()
        # 將test值加入到ARIMA已預測之後的值
        model_fit = model_fit.append(test1[i:i+1], refit=False)
        # 將預測值存起來
        predictions_ARIMA.append(pred[0])
    # 小於0的話就轉換等於0
    predictions_ARIMA = [p if p>=0 else 0 for p in predictions_ARIMA]


    ##### SingleLSTM #####

    # 取之前存的最佳參數
    df = pd.read_excel(f'{path}/Model/singleLSTM_test.xlsx',engine='openpyxl')
    # 之前的測試資料也只有部分有測試...所以如果目標的篩選不到就直接設定過去時間點為5
    try:
        days = list(df[(df['target'] == target) & (df['location'] == location)].sort_values('mae')['days'])[0]
    except:
        days = 5

    # 取得之前整理好的Training Data
    df = pd.read_csv(r"{}/Model/{}/{}.csv".format(path, target, location))
    # 刪除row值全部為1的column
    """
    後面這邊(x.startswith('total')) or (x in ['people_vaccinated', 'people_fully_vaccinated_per_hundred', 'people_fully_vaccinated', 'people_vaccinated_per_hundred'])
    在implement會解釋
    """
    c = list(df.columns)
    c.remove(target)
    for x in c:
        if (len(df[x].unique()) == 1) or (x.startswith('total')) or (x in ['people_vaccinated', 'people_fully_vaccinated_per_hundred', 'people_fully_vaccinated', 'people_vaccinated_per_hundred']):
            df.drop(x, 1, inplace=True)
    # 只取相關性大於0.3的欄位作為訓練資料
    df = df.loc[:, df.corr()[target]>0.3]
    if len(list(df))==0:
        return print('The data is all zero')
    # new_cases數值/10
    """
    implement解釋
    """
    if target == 'new_cases':
        df['new_cases'] = df['new_cases']/10

    # Split Training Data to train, val and test dataset        
    # 0.7作為訓練集
    train_size = round(len(df) * 0.7)
    # 0.1作為驗證集
    val_size = round(len(df) * 0.1)
    # 切分train, val, test
    train = df.iloc[:train_size, :]
    val = df.iloc[train_size:train_size+val_size, :]
    test = df.iloc[train_size+val_size:, :]

    # Preparing Data to LSTM Format
    X_train, Y_train = buildTrain_onetarget(train, target, days, 1)
    X_val, Y_val = buildTrain_onetarget(val, target, days, 1)
    X_test, Y_test_SingleLSTM = buildTrain_onetarget(test, target, days, 1)

    # shuffle the data
    X_train, Y_train = shuffle(X_train, Y_train)

    # Trasfer the data dimensions
    Y_train = Y_train[:, np.newaxis]
    Y_val = Y_val[:, np.newaxis]

    # Build Model and Training
    print('Training Single Feature LSTM Model...')
    # 建構模型
    model_SingleLSTM = buildOneToOneModel(X_train.shape)
    # 訓練模型
    model_SingleLSTM.fit(X_train, Y_train, epochs=500, batch_size=512, validation_data=(X_val, Y_val), verbose=0)

    # Predict
    predict_test_SigleLSTM = model_SingleLSTM.predict(X_test)

    # Transfer data dimensions for evaluation and plotting
    # new_cases的話要再X10回來，前面有除10
    if target == 'new_cases':
        Y_test_SingleLSTM = (Y_test_SingleLSTM.reshape(-1)*10).tolist()
        predict_test_SigleLSTM = (predict_test_SigleLSTM.reshape(-1)*10).tolist()
    else:
        Y_test_SingleLSTM = Y_test_SingleLSTM.reshape(-1).tolist()
        predict_test_SigleLSTM = predict_test_SigleLSTM.reshape(-1).tolist()

    # Remove predictinon result for lower then 0
    predict_test_SigleLSTM = [p if p>=0 else 0 for p in predict_test_SigleLSTM]


    ##### MultiLSTM #####

    # Get Best Parameter
    df = pd.read_excel(f'{path}/Model/LSTM_test.xlsx',engine='openpyxl')
    try:
        days = list(df[(df['target'] == target) & (df['location'] == location)].sort_values('mae')['days'])[0]
    except:
        days = 5

    for x in train.columns:
        if x != target:
            trans = MinMaxScaler()
            train[x] = trans.fit_transform(train[[x]])
            val[x] = trans.transform(val[[x]])
            test[x] = trans.transform(test[[x]])

    # Preparing Data to LSTM Format
    X_train, Y_train = buildTrain(train, target, days, 1)
    X_val, Y_val = buildTrain(val, target, days, 1)
    X_test, Y_test_LSTM = buildTrain(test, target, days, 1)

    # shuffle the data
    X_train, Y_train = shuffle(X_train, Y_train)

    # Trasfer the data dimensions
    Y_train = Y_train[:, np.newaxis]
    Y_val = Y_val[:, np.newaxis]

    # Build Model and Training
    print('Training Multi-Feature LSTM Model...')
    model_LSTM = buildManyToOneModel(X_train.shape)
    model_LSTM.fit(X_train, Y_train, epochs=500, batch_size=512, validation_data=(X_val, Y_val), verbose=0)

    # Predict
    predict_test_LSTM = model_LSTM.predict(X_test)

    # Transfer data dimensions for evaluation and plotting
    if target == 'new_cases':
        Y_test_LSTM = (Y_test_LSTM.reshape(-1)*10).tolist()
        predict_test_LSTM = (predict_test_LSTM.reshape(-1)*10).tolist()
    else:
        Y_test_LSTM = Y_test_LSTM.reshape(-1).tolist()
        predict_test_LSTM = predict_test_LSTM.reshape(-1).tolist()

    # Remove predictinon result for lower then 0
    predict_test_LSTM = [p if p>=0 else 0 for p in predict_test_LSTM]
    # 印出結果
    print('\n')
    print(f'##### Model Result on {location} - {target} #####')
    print('\n')
    # Print Evaluation Result
    # 每一個模型的結果分別計算mse, rmse, mae, r2
    d = [["ARIMA", mean_squared_error(test1, predictions_ARIMA, squared=False), mean_squared_error(test1, predictions_ARIMA), mean_absolute_error(test1, predictions_ARIMA), r2_score(test1, predictions_ARIMA)],
         ["Single Feature LSTM", mean_squared_error(Y_test_SingleLSTM, predict_test_SigleLSTM, squared=False), mean_squared_error(Y_test_SingleLSTM, predict_test_SigleLSTM), mean_absolute_error(Y_test_SingleLSTM, predict_test_SigleLSTM), r2_score(Y_test_SingleLSTM, predict_test_SigleLSTM)],
         ["Multi Feature LSTM", mean_squared_error(Y_test_LSTM, predict_test_LSTM, squared=False), mean_squared_error(Y_test_LSTM, predict_test_LSTM), mean_absolute_error(Y_test_LSTM, predict_test_LSTM), r2_score(Y_test_LSTM, predict_test_LSTM)]]

    print ("{:<30} {:<15} {:<15} {:<15} {:<15}".format('Model Name', 'RMSE', 'MSE', 'MAE', 'R-Square'))
    print("-"*90)
    for v in d:
        model_name, rmse, mse, mae, r2 = v
        print ("{:<31} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(model_name, rmse, mse, mae, r2))
    print("-"*90)

    # Model Saving
    # 挑選MAE最低的模型，只儲存最好的，檔名會儲存使用的參數，ex: BestModel_Japan_New Cases_ARIMA.pkl
    mae = [i[3] for i in d]
    print(f"Saving Best Performance Model {d[np.array(mae).argmin()][0]}")
    if d[np.array(mae).argmin()][0] == 'ARIMA':
        model_fit.save(f'../Model/BestModel_{location}_{target}_ARIMA.pkl')
    elif d[np.array(mae).argmin()][0] == "Single Feature LSTM":
        model_SingleLSTM.save(f'../Model/BestModel_{location}_{target}_SingleLSTM.h5')
    else:
        model_LSTM.save(f'../Model/BestModel_{location}_{target}_LSTM.h5')
        
    if target == 'new_deaths':
        target = 'deaths'
    elif target == 'new_cases':
        taregt = 'cases'
    else:
        target = 'ICU Patients'
    # Plotting
    fontsize=40
    # 設定20x15的子圖，並編輯第三個
    plt.figure(figsize=(40,20), dpi=300)
    # 實際值折線圖
    plt.plot(test1[-105:-5], 'g-', label='real', linewidth=3.0)
    # 預測值折線圖
    plt.plot(predictions_ARIMA[-105:-5], 'r-', label='predict', linewidth=3.0)
    # y刻度字大小
    plt.yticks(fontsize=fontsize)
    # x刻度字大小
    plt.xticks(fontsize=fontsize)
    # x標籤
    plt.xlabel('Day', fontsize=fontsize)
    # y標籤
    plt.ylabel(f'Daily new {target}', fontsize=fontsize)
    # 圖標題
    plt.title(f'Daily new {target} in {location} - ARIMA', fontsize=fontsize)
    # 圖示
    plt.legend(prop={'size': 40})
    plt.show()

    # 設定20x15的子圖，並編輯第三個
    plt.figure(figsize=(40,20), dpi=300)
    # 實際值折線圖
    plt.plot(Y_test_SingleLSTM[-105:-5], 'g-', label='real', linewidth=3.0)
    # 預測值折線圖
    plt.plot(predict_test_SigleLSTM[-105:-5], 'r-', label='predict', linewidth=3.0)
    # y刻度字大小
    plt.yticks(fontsize=fontsize)
    # x刻度字大小
    plt.xticks(fontsize=fontsize)
    # x標籤
    plt.xlabel('Day', fontsize=fontsize)
    # y標籤
    plt.ylabel(f'Daily new {target}', fontsize=fontsize)
    # 圖標題
    plt.title(f'Daily new {target} in {location} - Single Feature LSTM', fontsize=fontsize)
    # 圖示
    plt.legend(prop={'size': 40})
    plt.show()

    # 設定20x15的子圖，並編輯第三個
    plt.figure(figsize=(40,20), dpi=300)
    # 實際值折線圖
    plt.plot(Y_test_LSTM[-105:-5], 'g-', label='real', linewidth=3.0)
    # 預測值折線圖
    plt.plot(predict_test_LSTM[-105:-5], 'r-', label='predict', linewidth=3.0)
    # y刻度字大小
    plt.yticks(fontsize=fontsize)
    # x刻度字大小
    plt.xticks(fontsize=fontsize)
    # x標籤
    plt.xlabel('Day', fontsize=fontsize)
    # y標籤
    plt.ylabel(f'Daily new {target}', fontsize=fontsize)
    # 圖標題
    plt.title(f'Daily new {target} in {location} - Muti Feature LSTM', fontsize=fontsize)
    # 圖示
    plt.legend(prop={'size': 40})
    plt.show()