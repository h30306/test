import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from ipywidgets import interact
"""
檔案所在之位置
"""
path = '../Data'

# Jupyter notebook上顯示下拉式選單
@interact(Outcome=['case', 'death'], Product=['all_types', 'Janssen', 'Moderna', 'Pfizer'], Age=['all_ages', '12-17', '18-49', '50-64', '65+'])
# 定義使用function直接import到主程式當中
def DataVisualizationVaccineType(Outcome, Product, Age):
    df = pd.read_csv(f'{path}/Rates_of_COVID-19_Cases_or_Deaths_by_Age_Group_and_Vaccination_Status_and_Booster_Dose.csv')
    
    # 原有資料沒有日期，只有周，因此要加上日子，兩個outcome要分開跑
    df1 = df[df['outcome'] == 'case']
    df2 = df[df['outcome'] == 'death']
    
    d = []
    #加入日期21, 28, 7, 14
    r = ['21', '28', '7', '14']
    w = '38'
    index = 0

    for i in df1['mmwr_week']:
        if str(i)[-2:] == '52':
            d.append('31')
        elif str(i)[-2:] == w:
            d.append(r[index])
        else:
            index+=1
            w = str(i)[-2:]
            if index==4:
                index=0
            d.append(r[index])
    df1['day'] = d

    d = []
    r = ['21', '28', '7', '14']
    w = '38'
    index = 0

    for i in df2['mmwr_week']:
        if str(i)[-2:] == '52':
            d.append('31')
        elif str(i)[-2:] == w:
            d.append(r[index])
        else:
            index+=1
            w = str(i)[-2:]
            if index==4:
                index=0
            d.append(r[index])
    df2['day'] = d

    # 處理完再組合起來
    df = pd.concat([df1, df2], ignore_index=True)
    # 篩選對應的資料範圍
    temp = df[(df['outcome'] == Outcome) & (df['vaccine_product'] == Product) & (df['age_group'] == Age)]
    # 篩選完沒資料return沒資料
    if len(temp) == 0:
        return print('No data matching this filter')
    # 將日期轉換成年份跟月份
    temp['year'] = temp['mmwr_week'].apply(lambda x: str(x)[:4])
    temp['month'] = temp['month'].apply(lambda x: str(x)[:2])
    # 組成日期
    temp['date'] = temp.apply(lambda x : x['year']+'-'+x['month']+'-'+x['day'], axis=1)
    # 建構一個新的資料表
    new = pd.DataFrame()
    new['Date'] = temp['date']
    new['Date'] = pd.to_datetime(new['Date'])
    # 轉換各欄位為百分比
    new['Boosted With Outcome'] = (temp['boosted_with_outcome']/temp['boosted_population'])*100
    new['Vaccinated With Outcome'] = (temp['vaccinated_with_outcome']/temp['fully_vaccinated_population'])*100
    new['Unvaccinated With Outcome'] = (temp['unvaccinated_with_outcome']/temp['unvaccinated_population'])*100
    ### 以下為建構雙label的x軸，要有月份資料跟年份資料 ###
    tp_pass = [datetime.strptime(i, '%Y-%m-%d') for i in new['Date'].dt.strftime('%Y-%m-%d')]
    # 定義第一條線的資料
    azip_pass1 = new['Boosted With Outcome'].values
    # 定義第二條線的資料
    azip_pass2 = new['Vaccinated With Outcome'].values
    # 定義第三條線的資料
    azip_pass3 = new['Unvaccinated With Outcome'].values
    # 因為要畫雙label軸，所以把圖跟軸分開，並定義圖size 30x15
    fig, ax=plt.subplots(1, 1, figsize=(30, 15))
    # 畫第二條紅線，寬度5
    ax.plot(tp_pass, azip_pass1, color='r', linewidth=5.0)
    # 畫第二條藍線，寬度5
    ax.plot(tp_pass, azip_pass2, color='b', linewidth=5.0)
    # 畫第二條綠線，寬度5
    ax.plot(tp_pass, azip_pass3, color='g', linewidth=5.0)

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
    # 不同的圖圖標題不一樣
    if Outcome == 'case':
        plt.title(f'Distribution of people inflected COVID-19 after vaccination', fontsize=50)
    else:
        plt.title(f'Distribution of people dead caused by COVID-19 after vaccination', fontsize=50)
    # y標籤
    if Outcome == 'death':
        plt.ylabel('Percentage of people dead caused by COVID-19(%)', fontsize=40)
    else:
        plt.ylabel('Percentage of people inflected COVID-19(%)', fontsize=40)
    # 圖示
    ax.legend(['Third vaccine dose', 'Second vaccine dose', 'First vaccine dose'], loc='upper left', fontsize=30)
    # 顯示橫線
    plt.grid(1)
    plt.savefig('1.png', bbox_inches="tight", dpi=300)
    plt.show()