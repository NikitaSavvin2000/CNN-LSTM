
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Загрузка данных
df_Pl = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRKJzXJlQSNew0dxrQ8mFQMhBv_4owvfsF2If1b-rmxMZkR5gabHC4OiaSwt8Ul1Omc8taR27UohSeg/pub?gid=429486365&single=true&output=csv')

end_know_date = df_Pl['time'].iloc[-1]

end_know_date_index = df_Pl[df_Pl['time'] == end_know_date].index[0] +1

nan_indices = df_Pl.index[df_Pl['P_l'].isna()]

avg_pl = df_Pl['P_l'].mean()

end_date = df_Pl['time'].iloc[-1]

future_dates = pd.date_range(start='2023-09-10', end='2024-12-31', freq='5T')  # Измените freq по необходимости

future_df = pd.DataFrame({'time': future_dates, 'Scaled_P_l': np.nan})

df_Pl = pd.concat([df_Pl, future_df], ignore_index=True)

end_future_date = df_Pl['time'].iloc[-1]

df_Pl['P_l'].fillna(avg_pl, inplace=True)

df_Pl['time'] = pd.to_datetime(df_Pl['time'])

scaler = MinMaxScaler(feature_range=(0, 1))

df_Pl['Scaled_P_l'] = scaler.fit_transform(df_Pl['P_l'].values.reshape(-1, 1))

look_back = 300

end_date_index = df_Pl[df_Pl['time'] >= end_date].index.min()

X, y = [], []

for i in tqdm(range(end_date_index - look_back)):
    X.append(df_Pl[['Scaled_P_l', 'time']][i:(i + look_back)].values)
    y.append(df_Pl['Scaled_P_l'][i + look_back])

X, y = np.array(X), np.array(y)


from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, concatenate, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
import numpy as np

time_scaler = MinMaxScaler()

print(end_know_date_index)

train_size_percent = 0.95
train_size = int(end_know_date_index * train_size_percent)

# Проверка, чтобы train_size не превышало индекс end_know_date
train_size = min(train_size, end_know_date_index)


X_train, X_test = X[0:train_size, :], X[train_size:len(X), :]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

# Создание входов
input_cnn = Input(shape=(X_train.shape[1], 1), name='input_cnn')
input_lstm = Input(shape=(X_train.shape[1], 1), name='input_lstm')

# Сверточный слой для признака P_l
conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_cnn)
pool1 = MaxPooling1D(pool_size=2)(conv1)
flat1 = Flatten()(pool1)

# Вход для времени
flat2 = Flatten()(input_lstm)

# Объединение выходов сверточного слоя и выходов для времени
merged = concatenate([flat1, flat2])

dense = Dense(units=50, activation='relu')(merged)

output = Dense(units=1, activation='linear')(dense)

model = Model(inputs=[input_cnn, input_lstm], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

X_cnn = X_train[:, :, 0:1].astype('float32')

X_lstm = np.reshape(X_train[:, :, 1], (X_train.shape[0] * X_train.shape[1], 1))
X_lstm = np.vectorize(lambda x: x.timestamp())(X_train[:, :, 1]).astype('float32')
X_lstm_scaled = time_scaler.fit_transform(X_lstm)

X_lstm = np.reshape(X_lstm_scaled, (X_train.shape[0], X_train.shape[1], 1))

end_know_date_index = df_Pl[df_Pl['time'] == end_know_date].index[0]
X_train_cnn = X_cnn[:end_know_date_index]
X_train_lstm = X_lstm[:end_know_date_index]
y_train = y[:end_know_date_index]


epochs = 25
batch_size = 252

model.fit([X_train_cnn, X_train_lstm], y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)


def predict_consumption(dates_to_predict, model, X_cnn, df_Pl, scaler, look_back):
    dates_to_predict = pd.to_datetime(dates_to_predict)

    if dates_to_predict.tzinfo is None:
        dates_to_predict = dates_to_predict.tz_localize(df_Pl['time'].dt.tz)

    X_predict_cnn = []
    X_predict_lstm = []
    print(X_cnn)
    for date in dates_to_predict:
        # Получение последних look_back точек для каждой даты
        indices = df_Pl[df_Pl['time'] <= date].index[-look_back:]
        print(indices)
        X_predict_cnn.append(X_cnn[indices])
        X_predict_lstm.append(df_Pl[df_Pl['time'] <= date].tail(look_back)[['time']].values)

    X_predict_cnn = np.array(X_predict_cnn)
    X_predict_lstm = np.array(X_predict_lstm)

    X_predict_cnn = np.reshape(X_predict_cnn, (X_predict_cnn.shape[0], X_predict_cnn.shape[1], 1))

    # Преобразование времени
    X_predict_lstm[:,:,0] = (
        (X_predict_lstm[:,:,0].astype('datetime64[ns]').view('int64') -
         df_Pl['time'].min().timestamp()) /
        (df_Pl['time'].max().timestamp() - df_Pl['time'].min().timestamp())
    )

    predicted_scaled_values = model.predict([X_predict_cnn.astype('float32'), X_predict_lstm.astype('float32')])

    predicted_values = scaler.inverse_transform(predicted_scaled_values)

    df_predict = pd.DataFrame({'time': dates_to_predict, 'P_l': predicted_values.flatten()})

    return df_predict



import matplotlib.pyplot as plt


dates_to_predict = pd.date_range(start='2023-09-08', end='2023-09-13', freq='5T')
dates_to_predict = pd.to_datetime(dates_to_predict)

# google_drive_path = '/content/gdrive/MyDrive/'
# model_path = google_drive_path + f'model_e_{epochs}_bs_{batch_size}.h5'

# model =  load_model(model_path)
model = model

predicted_df = predict_consumption(dates_to_predict, model, X_cnn, df_Pl, scaler, look_back)

print(predicted_df)

plt.figure(figsize=(14, 7))
plt.plot(predicted_df['time'], predicted_df['P_l'], linewidth=1)
plt.title(label="Predict")
plt.ylabel("P_l Value")
plt.xlabel("Time")
plt.show()