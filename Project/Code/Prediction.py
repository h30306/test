import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings
from ipywidgets import interact
#避免產生警告等不必要訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')
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

# Jupyter notebook上顯示下拉式選單
@interact(target=['New Deaths', 'New Cases', 'ICU Patients'], location=['Austria', 'Australia', 'Belgium', 'Brazil', 'Canada', 'Denmark', 'France', 'Finland', 'Germany', 'Italy', 'Japan', 'Netherlands', 'New Zealand', 'Philippines', 'Poland', 'South Africa', 'South Korea', 'Turkey', 'United States', 'United Kingdom'])
# 定義使用function直接import到主程式當中
def Prediction(target, location):
    # 轉換目標內容，顯示的是New Deaths但欄位名稱是new_deaths
    if target == 'New Deaths': target = 'new_deaths'
    elif target == 'New Cases': target = 'new_cases'
    else: target = 'icu_patients'

    # 篩選最佳模型
    model_list = os.listdir('../Model')
    # 用檔名作為篩選條件
    model_name = [i for i in model_list if (target in i) & (location in i)]

    # 沒有篩選到的話，代表沒有訓練過，有的話才要做預測
    if len(model_name)==0:
        return print('Please Training the Model above before Evaluation')
    else:
        # 讀取資料
        df = pd.read_csv(r"{}/Model/{}/{}.csv".format(path, target, location))
        # 讀取模型資料
        print(f"Loading Best Model of prediction {target} of {location}: {model_name[0].split('_')[-1].split('.')[0]} Model")
        # 取得模型名稱，會有不同的資料建構法
        if model_name[0].split('_')[-1].split('.')[0] == 'ARIMA':
            # 讀取ARIMA模型
            model = ARIMAResults.load(f'../Model/{model_name[0]}')
            # 建構測試資料並且預測
            data = df[target].values
            # 使用ARIMA模型
            model = ARIMA(data, order=(5, 1, 2))
            model = model.fit()

            # 預測
            predictions = []
            for i in range(len(df[target])):
                pred = model.forecast()
                model = model.append(data[i:i+1], refit=False)
                predictions.append(pred[0])
        else:
            # 建構預測資料
            train = df[[target]]
            # 讀取模型
            model = load_model(f'../Model/{model_name[0]}')
            # 判斷模型類別
            if model_name[0].split('_')[-1].split('.')[0] == 'SigleLSTM':
                # 取之前存的最佳參數
                df = pd.read_excel(f'{path}/Model/singleLSTM_test.xlsx',engine='openpyxl')
                # 之前的測試資料也只有部分有測試...所以如果目標的篩選不到就直接設定過去時間點為5
                try:
                    days = list(df[(df['target'] == target) & (df['location'] == location)].sort_values('mae')['days'])[0]
                except:
                    days = 5
                # 建構測試資料
                X_test, data = buildTrain_onetarget(train, target, days, 1)
                # 預測
                predictions = model.predict(X_test)
            else:
                # Get Best Parameter
                df = pd.read_excel(f'{path}/Model/LSTM_test.xlsx',engine='openpyxl')
                try:
                    days = list(df[(df['target'] == target) & (df['location'] == location)].sort_values('mae')['days'])[0]
                except:
                    days = 5
                # 建構測試資料
                X_test, data = buildTrain(train, target, days, 1)
                # 預測
                predictions = model.predict(X_test)
        # 小於0的話就轉換等於0
        predictions = [p if p>=0 else 0 for p in predictions]

    # Save data
    pd.DataFrame({'days':[f'Day{i}' for i in range(1, len(data)+1)],'real value':data.ravel(), 'predict valus':[round(float(i)) for i in predictions]}).to_csv(f'../Data/Model_output/{location}_{target}_predict.csv', encoding='utf-8-sig', index=False)
    # Plotting
    # 計算mse
    mse = mean_squared_error(predictions, data)
    # 計算mae
    mae = mean_absolute_error(predictions, data)
    # 計算R-Square
    r_squared = r2_score(predictions, data)
    # 分隔線
    print('-'*100)
    print(f'Model Evaluation Result of {model_name[0].split("_")[-1].split(".")[0]} Model')
    # MSE只印到小數點後兩位
    print(f'(1) MSE of {model_name[0].split("_")[-1].split(".")[0]} Model: {mse:.2f}')
    # MAE只印到小數點後兩位
    print(f'(2) MAE of {model_name[0].split("_")[-1].split(".")[0]} Model: {mae:.2f}')
    # R-Square只印到小數點後兩位
    print(f'(3) R-squared of {model_name[0].split("_")[-1].split(".")[0]} Model: {r_squared:.2f}')
    # 分隔線
    print('-'*100)

    if target == 'new_deaths':
        target = 'deaths'
    elif target == 'new_cases':
        taregt = 'cases'
    else:
        target = 'ICU Patients'
    
    # 設定圖片10x5
    plt.figure(figsize=(10, 5), dpi=300)
    # 實際值折線圖
    plt.plot(data, 'g-', label='real')
    # 預測值折線圖
    plt.plot(predictions, 'r-', label='predict')
    # x標籤
    plt.xlabel('Day')
    # y標籤
    plt.ylabel(f'Daily new {target}')
    # 圖標題
    plt.title(f'Prediction of Daily new {target} in {location} - {model_name[0].split("_")[-1].split(".")[0]} Model')
    # 圖示
    plt.legend()
    # 顯示圖片
    plt.show()