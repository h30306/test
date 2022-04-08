import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from ipywidgets import interact
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings
#避免產生警告等不必要訊息
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows",1000) 
"""
檔案所在之位置
"""
path = '../Data'
"""
讀取df : 清整後的Part1 data
"""
df = pd.read_csv(f"{path}/corr_and_regression_data.csv")
# 轉換日期為datetime格式
df['Date'] = pd.to_datetime(df['Date'])

# Jupyter notebook上顯示下拉式選單
@interact(location=list(df['Location'].value_counts().keys().sort_values()), variable1=['New Cases', 'New Deaths', 'ICU Patients'], variable2=['People Vaccinated Per Hundred', 'People Full Vaccinated Per Hundred', 'Prevention Policy', 'Number of Detections Variant'], year=['all', '2020', '2021'], month=['all', 'Q1', 'Q2', 'Q3', 'Q4', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
# 定義使用function直接import到主程式當中
def DataAnalysis(location, variable1, variable2, year, month):
    ### 資料篩選 ###

    #判斷年的篩選，all的話就不用篩選年
    if year=='all':
        temp = df[df['Location'] == location]
    else:
        # 判斷月的篩選，all的話就不用篩選月
        if month=='all':
            # 篩選對應地點跟年
            temp = df[df['Location'] == location][(df['Date'].dt.year == int(year))]
        # 可以篩選Q幾
        elif month.startswith('Q'):
            # 挑選Q對應的月
            if month == 'Q1': monthes = [1,2,3]
            if month == 'Q2': monthes = [4,5,6]
            if month == 'Q3': monthes = [7,8,9]
            if month == 'Q4': monthes = [10,11,12]
            # 篩選對應地點跟年還有Q幾
            temp = df[df['Location'] == location][(df['Date'].dt.year == int(year)) & (df['Date'].dt.month>=monthes[0]) & (df['Date'].dt.month<=monthes[2])]
        # 不是Q的話就是月份
        else:
            # 篩選對應地點跟年還有月
            temp = df[df['Location'] == location][(df['Date'].dt.year == int(year)) & (df['Date'].dt.month == int(month))]
    # 如果variable2裡有Vaccinated，要篩選有值的資料
    if 'Vaccinated' in variable2:
        temp = temp[temp['People Vaccinated Per Hundred'] != float(0)]
    # variable1一定要篩選大於0的資料，其他是異常值
    temp = temp[temp[variable1]>0]
    # 篩選完沒資料return沒資料
    if len(temp)==0:
        return print('No data matching this filter')

    ### 篩選完開始建模 ###

    # Initial線性回歸模型
    model = LinearRegression()
    # training在variable1跟variable2
    model.fit(temp[variable1].values.reshape(-1,1), temp[variable2].values)
    # 計算MSE
    mse = np.mean((model.predict(temp[variable1].values.reshape(-1, 1)) - temp[variable1].values) ** 2)
    # 開根號=RMSE
    rmse = mse**0.5
    # 計算R-Square
    r_squared = model.score(temp[variable1].values.reshape(-1, 1), temp[variable2].values)
    # 計算調整R-Square
    adj_r_squared = r_squared - (1 - r_squared) * (temp[variable1].values.reshape(-1, 1).shape[1] / (temp[variable1].values.reshape(-1, 1).shape[0] - temp[variable1].values.reshape(-1, 1).shape[1] - 1))

    ### 印出模型績效 ###

    # 分隔線
    print('-'*100)
    # 取截距只印到小數點後兩位
    print(f'(1) Intercept of Linear Regression Model: {model.intercept_:.2f}')
    # 取回歸係數只印到小數點後兩位
    print(f'(2) Coefficient of Linear Regression Model: {model.coef_[0]:.2f}')
    # MSE只印到小數點後兩位
    print(f'(3) MSE of Linear Regression Model: {mse:.2f}')
    # RMSE只印到小數點後兩位
    print(f'(4) R-MSE of Linear Regression Model: {rmse:.2f}')
    # R-Square只印到小數點後兩位
    print(f'(5) R-squared of Linear Regression Model: {r_squared:.2f}')
    # 調整R-Square只印到小數點後兩位
    print(f'(6) Adjusted R-squared of Linear Regression Model: {adj_r_squared:.2f}')
    # 分隔線
    print('-'*100)

    fontsize=4
    # 設定圖片大小3x2
    plt.figure(figsize=(3,2), dpi=300)
    # variable1跟variable2的散點圖，顏色橘色
    plt.scatter(temp[variable2].values, temp[variable1].values,color='orange',s=7)
    # 用model預測結果來畫回歸線，顏色紅色, 粗細3.0
    plt.plot(model.predict(temp[variable1].values.reshape(-1, 1)), temp[variable1].values, color='r', linewidth=1.0)
    # x軸範圍-10~100
    plt.xlim(-10,100)
    # 圖標題
    plt.title(f'Regression Line of {variable1} and {variable2} in {location}', fontsize=fontsize)
    # y刻度字大小
    plt.yticks(fontsize=fontsize)
    # x刻度字大小
    plt.xticks(fontsize=fontsize)
    # y標籤
    plt.ylabel(variable1,fontsize=fontsize)
    # x標籤
    plt.xlabel(variable2,fontsize=fontsize)
    plt.show()

    # 設定圖片大小3x2
    plt.figure(figsize=(3,2), dpi=300)
    # 畫Correlation Best Fit Line
    sns.regplot(x=variable2, y=variable1, data=temp, scatter_kws={'s':2})
    # x軸範圍-10~100
    plt.xlim(-10,100)
    # 圖標題
    plt.title(f'Correlation Best Fit Line of {variable1} and {variable2} in {location}', fontsize=fontsize)
    # y刻度字大小
    plt.yticks(fontsize=fontsize)
    # x刻度字大小
    plt.xticks(fontsize=fontsize)
    # y標籤
    plt.ylabel(variable1,fontsize=fontsize)
    # x標籤
    plt.xlabel(variable2,fontsize=fontsize)
    # 顯示圖片
    plt.show()
    # 分隔線
    print('-'*100)
    # 直接計算相關係數
    print(f'(1) Correlation Between Two Variables: {temp[[variable1, variable2]].corr().values[1,0]:.2f}')
    # 分隔線
    print('-'*100)
