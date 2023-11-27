{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Задаем дату до которой теоритически будем прогнозировать\n",
    "end_future_date = '2024-12-31 23:55:00+00:00'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Читаем данные на которых будем обучать модель\n",
    "df_Pl = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRKJzXJlQSNew0dxrQ8mFQMhBv_4owvfsF2If1b-rmxMZkR5gabHC4OiaSwt8Ul1Omc8taR27UohSeg/pub?gid=429486365&single=true&output=csv')\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "outputs": [],
   "source": [
    "# Узнаем последнию известную дату. Нужна для создания промежутка значений будущих дат\n",
    "end_know_date = df_Pl['time'].iloc[-1]\n",
    "\n",
    "end_know_date_index = df_Pl[df_Pl['time'] == end_know_date].index[0] + 1\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Создаем общий датасет с известными и будущими данными нужно для корректного общего нормирования данных\n",
    "\n",
    "# Добавляем 5 минут чтобы последняя известная дата не попала в список для генерации будущих дат\n",
    "\n",
    "end_know_date = pd.to_datetime(end_know_date)\n",
    "\n",
    "# Добавление 5 минут\n",
    "end_know_date = end_know_date + pd.Timedelta(minutes=5)\n",
    "\n",
    "# Преобразование обратно в строку\n",
    "end_know_date = end_know_date.strftime('%Y-%m-%d %H:%M:%S%z')\n",
    "\n",
    "future_dates = pd.date_range(start=end_know_date, end=end_future_date, freq='5T')\n",
    "\n",
    "future_df = pd.DataFrame({'time': future_dates, 'P_l': np.nan})\n",
    "\n",
    "avg_pl = df_Pl['P_l'].mean()\n",
    "\n",
    "# Заполняем Nan если они есть в датафрейме с известными данными\n",
    "df_Pl['P_l'].fillna(avg_pl, inplace=True)\n",
    "\n",
    "# Создаем общий датаферйм с известными и будущими датами для корректной нормировки данных относительно существующих и будущих дат\n",
    "df_Pl_sacler = pd.concat([df_Pl, future_df], ignore_index=True)\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "outputs": [],
   "source": [
    "# Функция нормировки дат относительно всех дат(известных и которых хотим прогнощировать)\n",
    "\n",
    "def normalized_df(df, scaler):\n",
    "# Преобразование времени в формат datetime\n",
    "  df['time'] = pd.to_datetime(df['time'])\n",
    "  # Создание новых столбцов\n",
    "  df['year'] = df['time'].dt.year\n",
    "  df['month'] = df['time'].dt.month\n",
    "  df['day'] = df['time'].dt.day\n",
    "  df['hour'] = df['time'].dt.hour\n",
    "  df['minute'] = df['time'].dt.minute\n",
    "\n",
    "  # Удаление исходной колонки времени\n",
    "  df = df.drop('time', axis=1)\n",
    "\n",
    "  # Нормализация данных\n",
    "  df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)\n",
    "  return df_normalized\n",
    "\n",
    "# Обратное масштабирование данных времени\n",
    "def restored_date(df_normalized, scaler):\n",
    "  df_restored = pd.DataFrame(scaler.inverse_transform(df_normalized), columns=df_normalized.columns)\n",
    "  df_restored['time'] = pd.to_datetime(df_restored[['year', 'month', 'day', 'hour', 'minute']])\n",
    "  df_restored = df_restored.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)\n",
    "  return df_restored"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "# Нормализация всех данных существующих и будущих\n",
    "scaler = MinMaxScaler(feature_range=(0, 1))\n",
    "\n",
    "df_Pl_normalized_all_dates = normalized_df(df_Pl_sacler, scaler)\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "outputs": [],
   "source": [
    "# Данные для обучения отнормированные относительно всего промежутка(известных дат и дат для прогноза)\n",
    "\n",
    "df_Pl_normalized_initial_dates = df_Pl_normalized_all_dates.loc[: end_know_date_index-1]\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "outputs": [],
   "source": [
    "df_Pl_normalized_future_dates = df_Pl_normalized_all_dates.loc[end_know_date_index :]\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "outputs": [],
   "source": [
    "# Данные для обучения\n",
    "X_initial = df_Pl_normalized_initial_dates[['year', 'month', 'day', 'hour', 'minute']].values\n",
    "y_initial = df_Pl_normalized_initial_dates['P_l'].values\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "outputs": [],
   "source": [
    "\n",
    "# gреобразование в 5 мерный массив\n",
    "X_initial_train = X_initial.reshape(-1, 1, 5)\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "outputs": [
    {
     "data": {
      "text/plain": "array([[[0.        , 0.45454545, 0.3       , 0.69565217, 0.90909091]],\n\n       [[0.        , 0.45454545, 0.3       , 0.69565217, 1.        ]],\n\n       [[0.        , 0.45454545, 0.3       , 0.73913043, 0.        ]],\n\n       ...,\n\n       [[0.875     , 0.72727273, 0.33333333, 0.60869565, 0.90909091]],\n\n       [[0.875     , 0.72727273, 0.33333333, 0.60869565, 1.        ]],\n\n       [[0.875     , 0.72727273, 0.33333333, 0.65217391, 0.        ]]])"
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_initial_train"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Данные для обучения\n",
    "X_initial = df_Pl_normalized_initial_dates[['year', 'month', 'day', 'hour', 'minute']].values\n",
    "y_initial = df_Pl_normalized_initial_dates['P_l'].values\n",
    "\n",
    "# Преобразование в 5-мерный массив\n",
    "X_initial_train = X_initial.reshape(-1, 1, 5)\n",
    "\n",
    "# Разделение данных на обучающий и тестовый наборы\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_initial_train, y_initial, test_size=0.01, random_state=42)\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(755262, 1, 5)\n",
      "KerasTensor(type_spec=TensorSpec(shape=(None, 1, 5), dtype=tf.float32, name='input_cnn'), name='input_cnn', description=\"created by layer 'input_cnn'\")\n",
      "Epoch 1/10\n",
      "1476/1476 [==============================] - 4s 2ms/step - loss: 0.0032\n",
      "Epoch 2/10\n",
      "1476/1476 [==============================] - 4s 2ms/step - loss: 0.0022\n",
      "Epoch 3/10\n",
      "1476/1476 [==============================] - 3s 2ms/step - loss: 0.0019\n",
      "Epoch 4/10\n",
      "1476/1476 [==============================] - 3s 2ms/step - loss: 0.0018\n",
      "Epoch 5/10\n",
      "1476/1476 [==============================] - 4s 2ms/step - loss: 0.0016\n",
      "Epoch 6/10\n",
      "1476/1476 [==============================] - 4s 2ms/step - loss: 0.0015\n",
      "Epoch 7/10\n",
      "1476/1476 [==============================] - 4s 2ms/step - loss: 0.0015\n",
      "Epoch 8/10\n",
      "1476/1476 [==============================] - 4s 2ms/step - loss: 0.0014\n",
      "Epoch 9/10\n",
      "1476/1476 [==============================] - 4s 2ms/step - loss: 0.0014\n",
      "Epoch 10/10\n",
      "1476/1476 [==============================] - 3s 2ms/step - loss: 0.0014\n"
     ]
    },
    {
     "data": {
      "text/plain": "<keras.src.callbacks.History at 0x141fdce10>"
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, concatenate\n",
    "from tensorflow.keras.models import Model\n",
    "\n",
    "print(X_train.shape)\n",
    "# Создание входов\n",
    "input_cnn = Input(shape=(1, 5), name='input_cnn')\n",
    "input_lstm = Input(shape=(1, 5), name='input_lstm')\n",
    "print(input_cnn)\n",
    "# Сверточный слой для признака P_l\n",
    "conv1 = Conv1D(filters=32, kernel_size=1, activation='relu')(input_cnn)  # изменено kernel_size на 1\n",
    "pool1 = MaxPooling1D(pool_size=1)(conv1)\n",
    "flat1 = Flatten()(pool1)\n",
    "\n",
    "# LSTM слой для времени\n",
    "lstm = LSTM(units=50, activation='relu')(input_lstm)\n",
    "\n",
    "# Объединение выходов сверточного слоя и выходов для времени\n",
    "merged = concatenate([flat1, lstm])\n",
    "\n",
    "dense = Dense(units=50, activation='relu')(merged)\n",
    "\n",
    "output = Dense(units=1, activation='linear')(dense)\n",
    "\n",
    "model = Model(inputs=[input_cnn, input_lstm], outputs=output)\n",
    "model.compile(optimizer='adam', loss='mean_squared_error')\n",
    "\n",
    "# Обучение модели\n",
    "epochs = 10\n",
    "batch_size = 512\n",
    "model.fit([X_train, X_train], y_train, epochs=epochs, batch_size=batch_size)\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "def get_indices_between_dates(df_Pl_sacler, start_date, end_date):\n",
    "    \"\"\"\n",
    "    Получает DataFrame, начальную и конечную даты, возвращает список индексов,\n",
    "    соответствующих датам в заданном диапазоне.\n",
    "    Нужна для получения индексов времени которые мы хотим спрогнозировать\n",
    "    Parameters:\n",
    "    - df_Pl_sacler: pandas.DataFrame\n",
    "        DataFrame с колонкой 'time', содержащей даты в формате 'yyyy-mm-dd hh:mm:ss+00:00'.\n",
    "    - start_date: str\n",
    "        Начальная дата в формате 'yyyy-mm-dd'.\n",
    "    - end_date: str\n",
    "        Конечная дата в формате 'yyyy-mm-dd'.\n",
    "\n",
    "    Returns:\n",
    "    - list\n",
    "        Список индексов, соответствующих датам в заданном диапазоне.\n",
    "    \"\"\"\n",
    "    # Преобразование колонки 'time' в формат datetime\n",
    "    df_Pl_sacler['time'] = pd.to_datetime(df_Pl_sacler['time'])\n",
    "\n",
    "    # Выборка индексов для дат в заданном диапазоне\n",
    "    indices = df_Pl_sacler.loc[(df_Pl_sacler['time'] >= start_date) & (df_Pl_sacler['time'] <= end_date)].index\n",
    "\n",
    "    return indices\n",
    "\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "outputs": [],
   "source": [
    "start_date = '2023-09-05'\n",
    "end_date = '2023-09-16'"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "outputs": [],
   "source": [
    "'''\n",
    "Блок получения нормированных дат для подстановки в модель\n",
    "'''\n",
    "\n",
    "# Даты на которые хотим сделать прогноз\n",
    "start_date = '2023-09-05'\n",
    "end_date = '2023-09-16'\n",
    "\n",
    "df_Pl_all_dates = df_Pl_sacler.copy()\n",
    "\n",
    "# выбираем индексы для которых хотим сделать прогноз из общего датасета(с бущими и настоящими датами)\n",
    "indexes = get_indices_between_dates(df_Pl_all_dates, start_date, end_date)\n",
    "\n",
    "# Нормализуем датасет чтобы получить нормализованные даты чтобы корректно поместить их в модель для предсказания\n",
    "df_Pl_all_dates_normilize = normalized_df(df_Pl_all_dates, scaler)\n",
    "\n",
    "# Получаем датафрейм с нормированными датами на промежуток прогноза\n",
    "df_Pl_sacler_for_predict = df_Pl_all_dates_normilize.loc[indexes]\n",
    "\n",
    "X_predict = df_Pl_sacler_for_predict[['year', 'month', 'day', 'hour', 'minute']].values\n"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "outputs": [
    {
     "data": {
      "text/plain": "numpy.ndarray"
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(X_predict)"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "outputs": [],
   "source": [],
   "metadata": {
    "collapsed": false
   }
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
