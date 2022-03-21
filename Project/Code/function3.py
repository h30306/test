import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
import warnings
from ipywidgets import interact
warnings.filterwarnings('ignore')

path = '../Data'

def buildTrain(train, target, pastDay=7, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay][target]))
    return np.array(X_train), np.array(Y_train)

def buildTrain_onetarget(train, target, pastDay=7, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay][[target]]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay][target]))
    return np.array(X_train), np.array(Y_train)

def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def splitData(X, Y, rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val

def buildOneToOneModel(shape):
    model = Sequential()
    model.add(LSTM(128, input_length=shape[1], input_dim=shape[2], return_sequences=False))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)
    model.compile(loss="mae", optimizer=adam)
    #model.summary()
    return model

def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(128, input_length=shape[1], input_dim=shape[2], return_sequences=False))
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)
    model.compile(loss="mae", optimizer=adam)
    #model.summary()
    return model
    
@interact(target=['New Deaths', 'New Cases', 'ICU Patients'], location=['Japan', 'Germany', 'Turkey', 'Canada', 'United Kingdom', 'South Korea', 'United States', 'South Africa'])
def Model_training(target, location):
    if target == 'New Deaths': target = 'new_deaths'
    elif target == 'New Cases': target = 'new_cases'
    else: target = 'icu_patients'
    ##### Arima #####

    # Get Best Parameter
    df = pd.read_excel(f'{path}/Model/ARIMA_test.xlsx',engine='openpyxl')
    try:
        p = int(df[(df['target'] == target) & (df['location'] == location)]['p'])
        q = int(df[(df['target'] == target) & (df['location'] == location)]['q'])
    except:
        p,q = 5, 2

    # Get Training Data
    df = pd.read_csv(r"{}/Model/{}/{}.csv".format(path, target, location))
    for x in df.columns:
        if len(df[x].unique()) == 1:
            df.drop(x, 1, inplace=True)

    # Split Training Data to train and test dataset        
    value = df[target].values # 目標變數
    length = int(len(value) * 0.7) # 訓練集筆數
    train1 = list(value[0:length]) # 訓練集
    test1 = list(value[int(len(value) * 0.8):]) # 測試集

    # Build Model and Training
    print('Training ARIMA Model...')
    model = ARIMA(train1, order=(p, 1, q))
    model_fit = model.fit()
    predictions_ARIMA = []
    for i in range(len(test1)):
        pred = model_fit.forecast()
        model_fit = model_fit.append(test1[i:i+1], refit=False)
        predictions_ARIMA.append(pred[0])

    predictions_ARIMA = [p if p>=0 else 0 for p in predictions_ARIMA]


    ##### SingleLSTM #####

    # Get Best Parameter
    df = pd.read_excel(f'{path}/Model/singleLSTM_test.xlsx',engine='openpyxl')
    try:
        days = list(df[(df['target'] == target) & (df['location'] == location)].sort_values('mae')['days'])[0]
    except:
        days = 5

    # Data Preprocessing
    df = pd.read_csv(r"{}/Model/{}/{}.csv".format(path, target, location))
    for x in df.columns:
        if (len(df[x].unique()) == 1) or (x.startswith('total')) or (x in ['people_vaccinated', 'people_fully_vaccinated_per_hundred', 'people_fully_vaccinated', 'people_vaccinated_per_hundred']):
            df.drop(x, 1, inplace=True)
    df = df.loc[:, df.corr()[target]>0.3]
    if target == 'new_cases':
        df['new_cases'] = df['new_cases']/10

    # Split Training Data to train, val and test dataset        
    train_size = round(len(df) * 0.7)
    val_size = round(len(df) * 0.1)
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
    model_SigleLSTM = buildOneToOneModel(X_train.shape)
    model_SigleLSTM.fit(X_train, Y_train, epochs=500, batch_size=512, validation_data=(X_val, Y_val), verbose=0)

    # Predict
    predict_test_SigleLSTM = model_SigleLSTM.predict(X_test)

    # Transfer data dimensions for evaluation and plotting
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
    print('\n')
    print(f'##### Model Result on {location} - {target} #####')
    print('\n')
    # Print Evaluation Result
    d = [ ["ARIMA", mean_squared_error(test1, predictions_ARIMA, squared=False), mean_squared_error(test1, predictions_ARIMA), mean_absolute_error(test1, predictions_ARIMA), r2_score(test1, predictions_ARIMA)],
         ["Single Feature LSTM", mean_squared_error(Y_test_SingleLSTM, predict_test_SigleLSTM, squared=False), mean_squared_error(Y_test_SingleLSTM, predict_test_SigleLSTM), mean_absolute_error(Y_test_SingleLSTM, predict_test_SigleLSTM), r2_score(Y_test_SingleLSTM, predict_test_SigleLSTM)],
         ["Multi Feature LSTM", mean_squared_error(Y_test_LSTM, predict_test_LSTM, squared=False), mean_squared_error(Y_test_LSTM, predict_test_LSTM), mean_absolute_error(Y_test_LSTM, predict_test_LSTM), r2_score(Y_test_LSTM, predict_test_LSTM)]]

    print ("{:<30} {:<15} {:<15} {:<15} {:<15}".format('Model Name', 'RMSE', 'MSE', 'MAE', 'R-Square'))
    print("-"*90)
    for v in d:
        model_name, rmse, mse, mae, r2 = v
        print ("{:<31} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(model_name, rmse, mse, mae, r2))
    print("-"*90)

    # Plotting
    plt.figure(figsize=(20, 5), dpi=300)
    plt.subplot(131)
    plt.plot(test1[-105:-5], 'g-', label='real')
    plt.plot(predictions_ARIMA[-105:-5], 'r-', label='predict')
    plt.xlabel('days')
    plt.ylabel(target)
    plt.title(f'{location} - {target} - ARIMA')
    plt.legend()

    plt.subplot(132)
    plt.plot(Y_test_SingleLSTM[-105:-5], 'g-', label='real')
    plt.plot(predict_test_SigleLSTM[-105:-5], 'r-', label='predict')
    plt.xlabel('days')
    plt.ylabel(target)
    plt.title(f'{location} - {target} - Single Feature LSTM')
    plt.legend()

    plt.subplot(133)
    plt.plot(Y_test_LSTM[-105:-5], 'g-', label='real')
    plt.plot(predict_test_LSTM[-105:-5], 'r-', label='predict')
    plt.xlabel('days')
    plt.ylabel(target)
    plt.title(f'{location} - {target} - Muti Feature LSTM')
    plt.legend()