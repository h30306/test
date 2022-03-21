import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from ipywidgets import interact
from sklearn.linear_model import LinearRegression
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

@interact(location=list(df['Location'].value_counts().keys().sort_values()), start=[f'20{y}{str(m).zfill(2)}' for y in range(20,23) for m in range(1,13)], end=[f'20{y}{str(m).zfill(2)}' for y in range(20,22) for m in range(1,13)], variable=['Vaccine', 'Variant'])
def time_line_with_variable(location, start, end, variable):
    temp = df[df['Location'] == location][(df['Date'].dt.year>=int(start[:4])) & (df['Date'].dt.year<=int(end[:4])) &(df['Date'].dt.month>=int(start[-2:])) & (df['Date'].dt.month<=int(end[-2:]))]
    if variable=='Vaccine':
        temp = temp[temp['People Vaccinated Per Hundred'] != float(0)]
        if len(temp)==0:
            return print('No data matching this filter')
        tp_pass = [datetime.strptime(i, '%Y-%m-%d') for i in temp['Date'].dt.strftime('%Y-%m-%d')]
        azip_pass1 = temp['People Vaccinated Per Hundred'].values
        azip_pass2 = temp['People Full Vaccinated Per Hundred'].values
        fig, ax=plt.subplots(1, 1, figsize=(30, 15))

        ax.plot(tp_pass, azip_pass1, color='r', linewidth=5.0)
        ax.plot(tp_pass, azip_pass2, color='b', linewidth=5.0)

        # Minor ticks every month.
        fmt_month = mdates.MonthLocator()
        # Minor ticks every year.
        fmt_year = mdates.YearLocator()

        ax.set_ylim(0, 100)
        ax.xaxis.set_minor_locator(fmt_month)
        # '%b' to get the names of the month
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(fmt_year)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        ax.tick_params(labelsize=30, which='both')
        sec_xaxis = ax.secondary_xaxis(-0.1)
        sec_xaxis.xaxis.set_major_locator(fmt_year)
        sec_xaxis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Hide the second x-axis spines and ticks
        sec_xaxis.spines['bottom'].set_visible(False)
        sec_xaxis.tick_params(length=0, labelsize=50)

        plt.yticks([0, 20, 40, 60, 80, 100], ["{}%".format(x) for x in [0, 20, 40, 60, 80, 100]], fontsize=30)
        plt.tight_layout()
        plt.title(f'Vaccination Rate (%) in {location}', fontsize=50)
        plt.ylabel('Vaccination Rate (%)', fontsize=40)
        ax.legend(['People Vaccinated Per Hundred', 'People Full Vaccinated Per Hundred'], loc='upper left', fontsize=30)
        plt.grid(1)
        plt.show()
    else:
        target_column = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron', 'Mu', 'SGTF', 'UNK', 'Other', 'Number of Detections Variant']
        temp = df[df['Location'] == location][(df['Date'].dt.year>=int(start[:4])) & (df['Date'].dt.year<=int(end[:4])) &(df['Date'].dt.month>=int(start[-2:])) & (df['Date'].dt.month<=int(end[-2:]))]
        temp = pd.concat([temp.iloc[:,:2], temp.iloc[:, 8:]], axis=1)
        temp.columns = temp.columns.map({'Location':'Location', 'Date':'Date', 'B.1.1.7':'Alpha', 'B.1.1.7+E484K':'Alpha', 'B.1.351':'Beta', 
                         'P.1':'Gamma', 'B.1.617.2':'Delta', 'B.1.1.529':'Omicron', 'B.1.621':'Mu', 'SGTF':'SGTF', 'UNK':'UNK',
                        'Number of Detections Variant':'Number of Detections Variant'}).fillna('Other')

        tp_pass = [datetime.strptime(i, '%Y-%m-%d') for i in temp['Date'].dt.strftime('%Y-%m-%d')]
        colors = ['b', 'g', 'c', 'm', 'y', 'k', 'orange', 'gold', 'tab:brown', 'r']
        fig, ax = plt.subplots(1, 1, figsize=(25, 20))
        for column, color in zip(target_column, colors*30):
            ax.plot(tp_pass, temp[[column]].sum(1).values, color=color, linewidth=7.0)

        # Minor ticks every month.
        fmt_month = mdates.MonthLocator()
        # Minor ticks every year.
        fmt_year = mdates.YearLocator()

        #ax.set_ylim(0, 5)
        ax.xaxis.set_minor_locator(fmt_month)
        # '%b' to get the names of the month
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(fmt_year)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        ax.tick_params(labelsize=30, which='both')
        sec_xaxis = ax.secondary_xaxis(-0.1)
        sec_xaxis.xaxis.set_major_locator(fmt_year)
        sec_xaxis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Hide the second x-axis spines and ticks
        sec_xaxis.spines['bottom'].set_visible(False)
        sec_xaxis.tick_params(length=0, labelsize=50)

        #plt.yticks([0, 20, 40, 60, 80, 100], ["{}%".format(x) for x in [0, 20, 40, 60, 80, 100]], fontsize=30)
        plt.tight_layout()
        plt.title(f'New Cases of Variant in {location}', fontsize=50)
        plt.ylabel('New Cases of Variant', fontsize=40)
        ax.legend(['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron', 'Mu', 'Other', 'UNK', 'SGTF', 'Summation of Detections Variant'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=30)
        plt.show()