
# Задаем дату до которой теоритически будем прогнозировать
end_future_date = '2024-12-31 23:55:00+00:00'

import pandas as pd

# Читаем данные на которых будем обучать модель
df_Pl = pd.read_csv('/Users/nikitasavvin/Desktop/Учеба/CNN-LSTM/cnn_lstm_notebooks/input_data/imputed_filled_P_l_LR (1).csv')

# Узнаем последнию известную дату. Нужна для создания промежутка значений будущих дат
end_know_date = df_Pl['time'].iloc[-1]

end_know_date_index = df_Pl[df_Pl['time'] == end_know_date].index[0] + 1

import numpy as np

# Создаем общий датасет с известными и будущими данными нужно для корректного общего нормирования данных

# Добавляем 5 минут чтобы последняя известная дата не попала в список для генерации будущих дат

end_know_date = pd.to_datetime(end_know_date)

# Добавление 5 минут
end_know_date = end_know_date + pd.Timedelta(minutes=5)

# Преобразование обратно в строку
end_know_date = end_know_date.strftime('%Y-%m-%d %H:%M:%S%z')

future_dates = pd.date_range(start=end_know_date, end=end_future_date, freq='5T')

future_df = pd.DataFrame({'time': future_dates, 'P_l': np.nan})

avg_pl = df_Pl['P_l'].mean()

# Заполняем Nan если они есть в датафрейме с известными данными
df_Pl['P_l'].fillna(avg_pl, inplace=True)

# Создаем общий датаферйм с известными и будущими датами для корректной нормировки данных относительно существующих и будущих дат
df_Pl_sacler = pd.concat([df_Pl, future_df], ignore_index=True)


# Функция нормировки дат относительно всех дат(известных и которых хотим прогнощировать)

def normalized_df(df, scaler):
# Преобразование времени в формат datetime
  df['time'] = pd.to_datetime(df['time'])
  # Создание новых столбцов
  df['year'] = df['time'].dt.year
  df['month'] = df['time'].dt.month
  df['day'] = df['time'].dt.day
  df['hour'] = df['time'].dt.hour
  df['minute'] = df['time'].dt.minute

  # Удаление исходной колонки времени
  df = df.drop('time', axis=1)

  # Нормализация данных
  df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
  return df_normalized

# Обратное масштабирование данных времени
def restored_date(df_normalized, scaler):
  df_restored = pd.DataFrame(scaler.inverse_transform(df_normalized), columns=df_normalized.columns)
  df_restored['time'] = pd.to_datetime(df_restored[['year', 'month', 'day', 'hour', 'minute']])
  df_restored = df_restored.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)
  return df_restored


from sklearn.preprocessing import MinMaxScaler

# Нормализация всех данных существующих и будущих
scaler = MinMaxScaler(feature_range=(0, 1))

df_Pl_normalized_all_dates = normalized_df(df_Pl_sacler, scaler)


# Данные для обучения отнормированные относительно всего промежутка(известных дат и дат для прогноза)

df_Pl_normalized_initial_dates = df_Pl_normalized_all_dates.loc[: end_know_date_index-1]


df_Pl_normalized_future_dates = df_Pl_normalized_all_dates.loc[end_know_date_index :]


# Данные для обучения
X_initial = df_Pl_normalized_initial_dates[['year', 'month', 'day', 'hour', 'minute']].values
y_initial = df_Pl_normalized_initial_dates['P_l'].values



# gреобразование в 5 мерный массив
X_initial_train = X_initial.reshape(-1, 1, 5)


from sklearn.model_selection import train_test_split

# Данные для обучения
X_initial = df_Pl_normalized_initial_dates[['year', 'month', 'day', 'hour', 'minute']].values
y_initial = df_Pl_normalized_initial_dates['P_l'].values

# Преобразование в 5-мерный массив
X_initial_train = X_initial.reshape(-1, 1, 5)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_initial_train, y_initial, test_size=0.01, random_state=42)



'''
Блок обучения модели
'''
#можно менять параметры активации количество слоев и прочее

from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.models import Model

# Создание входов
input_cnn = Input(shape=(1, 5), name='input_cnn')
input_lstm = Input(shape=(1, 5), name='input_lstm')
# Сверточный слой для признака P_l
conv1 = Conv1D(filters=Conv1D_filters, kernel_size=1, activation='tanh')(input_cnn)  # изменено kernel_size на 1
pool1 = MaxPooling1D(pool_size=1)(conv1)
flat1 = Flatten()(pool1)

lstm1 = LSTM(units=lstm1_units, activation='tanh', return_sequences=True)(input_lstm)
lstm2 = LSTM(units=lstm2_units, activation='sigmoid', return_sequences=True)(lstm1)
lstm3 = LSTM(units=lstm3_units, activation='tanh', return_sequences=True)(lstm2)
lstm4 = LSTM(units=lstm4_units, activation='tanh', return_sequences=True)(lstm3)
lstm5 = LSTM(units=lstm5_units, activation='tanh')(lstm4)

# Объединение выходов сверточного слоя и выходов для времени
merged = concatenate([flat1, lstm5])

dense = Dense(units=dense_units, activation='tanh')(merged)

output = Dense(units=1, activation='tanh')(dense)

model = Model(inputs=[input_cnn, input_lstm], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
epochs = epochs
batch_size = batch_size
model.fit([X_train, X_train], y_train, epochs=epochs, batch_size=batch_size)


'''
созадай из этого кода классы и методы для обучения модели нейронной сети
где как переменные должны быть 
end_future_date
ссылка на df_Pl = pd.read_csv
количество слоев их типы lstm cnn rnn их последовательность количество units каждом слое функции активации activation 
оптимизатор модели
количество epochs 
количество batch_size 
'''