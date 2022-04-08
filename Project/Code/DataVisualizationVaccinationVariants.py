import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from ipywidgets import interact
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
@interact(location=list(df['Location'].value_counts().keys().sort_values()), start=[f'20{y}{str(m).zfill(2)}' for y in range(20,22) for m in range(1,13)], end=[f'20{y}{str(m).zfill(2)}' for y in range(20,22) for m in range(1,13)], variable=['Vaccine', 'Variant'])
# 定義使用function直接import到主程式當中
def DataVisualizationVaccinationVariants(location, start, end, variable):
    ### 資料篩選 ###

    # 篩選對應的時間範圍
    temp = df[df['Location'] == location][(df['Date'].dt.year>=int(start[:4])) & (df['Date'].dt.year<=int(end[:4])) &(df['Date'].dt.month>=int(start[-2:])) & (df['Date'].dt.month<=int(end[-2:]))]
    # 如果變數是疫苗
    if variable=='Vaccine':
        # 要篩選有值的資料
        temp = temp[temp['People Vaccinated Per Hundred'] != float(0)]
        # 篩選完沒資料return沒資料
        if len(temp)==0:
            return print('No data matching this filter')
        ### 以下為建構雙label的x軸，要有月份資料跟年份資料 ###
        tp_pass = [datetime.strptime(i, '%Y-%m-%d') for i in temp['Date'].dt.strftime('%Y-%m-%d')]
        # 定義第一條線的資料
        azip_pass1 = temp['People Vaccinated Per Hundred'].values
        # 定義第二條線的資料
        azip_pass2 = temp['People Full Vaccinated Per Hundred'].values
        # 因為要畫雙label軸，所以把圖跟軸分開，並定義圖size 30x15
        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        # 畫第一條紅線，寬度5
        ax.plot(tp_pass, azip_pass1, color='r', linewidth=5.0)
        # 畫第二條藍線，寬度5
        ax.plot(tp_pass, azip_pass2, color='b', linewidth=5.0)

        ### 以下為建構月份x軸以及年份x軸 ###

        fmt_month = mdates.MonthLocator()
        fmt_year = mdates.YearLocator()
        ax.set_ylim(0, 100)
        ax.xaxis.set_minor_locator(fmt_month)
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(fmt_year)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(labelsize=30, which='both')
        sec_xaxis = ax.secondary_xaxis(-0.1)
        sec_xaxis.xaxis.set_major_locator(fmt_year)
        sec_xaxis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        sec_xaxis.spines['bottom'].set_visible(False)
        sec_xaxis.tick_params(length=0, labelsize=50)

        ### 建構月份x軸以及年份x軸done ###

        # y刻度顯示百分比
        plt.yticks([0, 20, 40, 60, 80, 100], ["{}%".format(x) for x in [0, 20, 40, 60, 80, 100]], fontsize=30)
        # 緊湊圖輸出
        plt.tight_layout()
        # 圖標題
        plt.title(f'Vaccination Rate (%) in {location}', fontsize=50)
        # y標籤
        plt.ylabel('Vaccination Rate (%)', fontsize=40)
        # 圖例
        ax.legend(['People Vaccinated Per Hundred', 'People Full Vaccinated Per Hundred'], loc='upper left', fontsize=30)
        # 顯示橫線
        plt.grid(1)

        # 顯示圖片
        plt.show()
    else:
        # 目標欄位
        target_column = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron', 'Mu', 'SGTF', 'UNK', 'Other']
        # 篩選對應的時間範圍
        temp = df[df['Location'] == location][(df['Date'].dt.year>=int(start[:4])) & (df['Date'].dt.year<=int(end[:4])) &(df['Date'].dt.month>=int(start[-2:])) & (df['Date'].dt.month<=int(end[-2:]))]
        # 挑選需要的欄位
        temp = pd.concat([temp.iloc[:,:2], temp.iloc[:, 8:]], axis=1)
        # 整理對應的病毒代號名稱
        temp.columns = temp.columns.map({'Location':'Location', 'Date':'Date', 'B.1.1.7':'Alpha', 'B.1.1.7+E484K':'Alpha', 'B.1.351':'Beta', 
                         'P.1':'Gamma', 'B.1.617.2':'Delta', 'B.1.1.529':'Omicron', 'B.1.621':'Mu', 'SGTF':'SGTF', 'UNK':'UNK'}).fillna('Other')
       # 篩選完沒資料return沒資料
        if len(temp)==0:
            return print('No data matching this filter')
        ### 以下為建構雙label的x軸，要有月份資料跟年份資料 ###
        tp_pass = [datetime.strptime(i, '%Y-%m-%d') for i in temp['Date'].dt.strftime('%Y-%m-%d')]
        # 定義不同病毒的顏色代號
        colors = ['b', 'g', 'c', 'm', 'y', 'k', 'orange', 'gold', 'tab:brown']
        # 因為要畫雙label軸，所以把圖跟軸分開，並定義圖size 25x20
        fig, ax = plt.subplots(1, 1, figsize=(25, 20))
        # 每一個column以及對應顏色畫圖
        for column, color in zip(target_column, colors):
            ax.plot(tp_pass, temp[[column]].sum(1).values, color=color, linewidth=7.0)

        ### 以下為建構月份x軸以及年份x軸 ###

        fmt_month = mdates.MonthLocator()
        fmt_year = mdates.YearLocator()
        ax.xaxis.set_minor_locator(fmt_month)
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(fmt_year)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(labelsize=30, which='both')
        sec_xaxis = ax.secondary_xaxis(-0.1)
        sec_xaxis.xaxis.set_major_locator(fmt_year)
        sec_xaxis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        sec_xaxis.spines['bottom'].set_visible(False)
        sec_xaxis.tick_params(length=0, labelsize=50)
        
        ### 建構月份x軸以及年份x軸done ###

        # 緊湊圖輸出
        plt.tight_layout()
        # 圖標題
        plt.title(f'New Cases of Variant in {location}', fontsize=50)
        # y標籤
        plt.ylabel('New Cases of Variant', fontsize=40)
        # 圖例子
        ax.legend(['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron', 'Mu', 'Other', 'UNK', 'SGTF', 'Summation of Detections Variant'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=30)
        # 顯示圖片
        plt.show()