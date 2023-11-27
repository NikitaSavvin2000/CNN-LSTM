import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Загрузка данных и предварительная обработка
# Замените 'your_data.csv' на путь к вашему файлу с данными
df = pd.read_csv('your_data.csv')
# Заменяем NaN значения на 0 (или другое подходящее начальное значение)
df['P_l'].fillna(0, inplace=True)

# Масштабирование данных
scaler = MinMaxScaler()
df['P_l'] = scaler.fit_transform(df['P_l'].values.reshape(-1, 1))

# Создание функции для создания последовательных данных
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append((sequence, target))
    return np.array(sequences)

# Определение параметров модели
seq_length = 10  # Длина последовательности
input_dim = 1  # Размерность входных данных (одна колонка P_l)
n_filters = 32  # Количество фильтров в сверточном слое
lstm_units = 64  # Количество нейронов в слое LSTM

# Создание последовательных данных
data = df['P_l'].values
sequences = create_sequences(data, seq_length)

# Разделение данных на обучающий и тестовый наборы
X = sequences[:, 0]
y = sequences[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение модели
model = Sequential()
model.add(Conv1D(filters=n_filters, kernel_size=3, activation='relu', input_shape=(seq_length, input_dim)))
model.add(LSTM(lstm_units))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Предсказание пропущенных значений
predicted_values = []
for i in range(len(data) - seq_length):
    sequence = data[i:i+seq_length]
    predicted_value = model.predict(np.array([sequence]))[0][0]
    predicted_values.append(predicted_value)

# Инверсия масштабирования
predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

# Замена NaN значений в исходных данных предсказанными значениями
df['P_l'].loc[:len(predicted_values) + seq_length - 1] = predicted_values

# Создание колонок с оригинальными значениями, заполненными значениями и всеми значениями
df['Original_P_l'] = scaler.inverse_transform(df['P_l'].values.reshape(-1, 1))
df['Filled_P_l'] = predicted_values
df['All_P_l'] = np.concatenate((df['Original_P_l'][:seq_length], predicted_values))

# Сохранение обновленных данных
df.to_csv('filled_data.csv', index=False)
