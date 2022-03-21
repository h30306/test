import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from ipywidgets import interact
from sklearn.linear_model import LinearRegression
import warnings
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
df['Date'] = pd.to_datetime(df['Date'])

@interact(location=list(df['Location'].value_counts().keys().sort_values()), variable1=['New Cases', 'New Deaths', 'ICU Patients'], variable2=['People Vaccinated Per Hundred', 'People Full Vaccinated Per Hundred', 'Prevention Policy', 'Number of Detections Variant'], year=['all', '2020', '2021'], month=['all', 'Q1', 'Q2', 'Q3', 'Q4', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
def scatter_and_correlation(location, variable1, variable2, year, month):
    if year=='all':
        temp = df[df['Location'] == location]
    else:
        if month=='all':
            temp = df[df['Location'] == location][(df['Date'].dt.year == int(year))]
        elif month.startswith('Q'):
            if month == 'Q1': monthes = [1,2,3]
            if month == 'Q2': monthes = [4,5,6]
            if month == 'Q3': monthes = [7,8,9]
            if month == 'Q4': monthes = [10,11,12]
            temp = df[df['Location'] == location][(df['Date'].dt.year == int(year)) & (df['Date'].dt.month>=monthes[0]) & (df['Date'].dt.month<=monthes[2])]
        else:
            temp = df[df['Location'] == location][(df['Date'].dt.year == int(year)) & (df['Date'].dt.month == int(month))]

    if 'Vaccinated' in variable2:
        temp = temp[temp['People Vaccinated Per Hundred'] != float(0)]
    temp = temp[temp[variable1]>0]
    if len(temp)==0:
        return print('No data matching this filter')

    model = LinearRegression()
    model.fit(temp[variable1].values.reshape(-1,1), temp[variable2].values)
    mse = np.mean((model.predict(temp[variable1].values.reshape(-1, 1)) - temp[variable1].values) ** 2)
    rmse = mse**0.5
    r_squared = model.score(temp[variable1].values.reshape(-1, 1), temp[variable2].values)
    adj_r_squared = r_squared - (1 - r_squared) * (temp[variable1].values.reshape(-1, 1).shape[1] / (temp[variable1].values.reshape(-1, 1).shape[0] - temp[variable1].values.reshape(-1, 1).shape[1] - 1))

    # 印出模型績效
    print('-'*100)
    print(f'(1) Correlation Between Two Variables: {temp[[variable1, variable2]].corr().values[1,0]:.5f}')
    print(f'(2) Intercept of Linear Regression Model: {model.intercept_:.5f}')
    print(f'(3) Coefficient of Linear Regression Model: {model.coef_[0]:.5f}')
    print(f'(4) MSE of Linear Regression Model: {mse:.5f}')
    print(f'(4) R-MSE of Linear Regression Model: {rmse:.5f}')
    print(f'(5) R-squared of Linear Regression Model: {r_squared:.5f}')
    print(f'(6) Adjusted R-squared of Linear Regression Model: {adj_r_squared:.5f}')
    print('-'*100)
    plt.figure(figsize=(6,4), dpi=300)
    plt.scatter(temp[variable2].values, temp[variable1].values,color='orange')
    plt.plot(model.predict(temp[variable1].values.reshape(-1, 1)), temp[variable1].values, color='r', linewidth=3.0)
    plt.xlim(-10,100)
    plt.title(f'Scatter Plot & Regression Line of {variable1} and {variable2}', fontsize=10)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel(variable1,fontsize=10)
    plt.xlabel(variable2,fontsize=10)
    plt.show()
