import sys
import pickle
import pandas as pd
import numpy as np
from glowbyte_transformers import DatetimeTransformer, FillNATransformer, GeneralizeConditionTransformer, OneHotEncoderTransformer, ResampleDataTransformer, HolidaysTransformer, CovidTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score#----------------------------


def process_dataset(dataset_path):
    # Здесь вы можете добавить код для обработки датасета
    print(f"Запуск прогнозирования для датасета по пути: {dataset_path}")
    
    try:
        # Загрузка модели из файла с использованием pickle
        with open('voting_model_imp.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            print("Модель успешно загружена:", loaded_model)
            train_df = pd.read_csv('train_test_df.csv')
            test_df = pd.read_csv(dataset_path)
            #Rename columns
            column_mapping = {'weather_пасмурно': 'cloudy', 'weather_снег': 'snow', 'weather_ясно':
                              'clear', 'weather_дождь/ливень/гроза': 'rain', 'weather_туман': 'fog'}
            #Pipeline steps
            pipeline = Pipeline(steps=[
                ('datetime_transform', DatetimeTransformer()),
                ('fill_na', FillNATransformer()),
                ('generalize_condition', GeneralizeConditionTransformer()),
                ('one_hot_encode', OneHotEncoderTransformer(columns=['weather'], column_mapping=column_mapping)),
                ('resample_data', ResampleDataTransformer()),
                ('holidays', HolidaysTransformer()),
                ('covid', CovidTransformer()),
            ])


            def make_features(dataset, column_name, max_lag, rolling_mean_size):
                dataset = dataset.copy()
                dataset['year'] = dataset['ds'].dt.year
                dataset['month'] = dataset['ds'].dt.month
                dataset['day'] = dataset['ds'].dt.day
                dataset['dayofweek'] = dataset['ds'].dt.dayofweek
                dataset['dayofyear'] = dataset['ds'].dt.dayofyear

                for lag in range(1, max_lag + 1):
                    dataset['lag_{}'.format(lag)] = dataset[column_name].shift(lag)

                dataset['rolling_mean'] = dataset[column_name].shift().rolling(rolling_mean_size).mean()
                return dataset   


            pipeline.fit(train_df)            
            train_transformed = pipeline.transform(train_df)
            test_transformed = pipeline.transform(test_df)

            data = pd.concat([train_transformed, test_transformed], ignore_index=True)
            data_f = make_features(data, 'target', 25, 25)

            # Определяем индекс, по которому разделить данные
            split_index = len(train_transformed)

            # Разделяем данные обратно на train_transformed и test_transformed
            train_transformed_back = data_f.iloc[:split_index, :]
            test_transformed_back = data_f.iloc[split_index:, :]
            train_transformed_back.set_index('ds', inplace=True)
            test_transformed_back.set_index('ds', inplace=True)

            X_train = train_transformed_back.drop(['target'], axis=1)
            y_train = train_transformed_back['target']
            X_test = test_transformed_back.drop(['target'], axis=1)
            y_test = test_transformed_back['target']

            voting_preds = loaded_model.predict(X_test)

            print(f'MAE: {mean_absolute_error(y_test, voting_preds)}')
            print(f'RMSE: {mean_squared_error(y_test, voting_preds, squared=False)}')
            print(f'MAPE: {mean_absolute_percentage_error(y_test, voting_preds)}')
            print(f'R2: {r2_score(y_test, voting_preds)}')

            res = test_transformed.copy()
            res = res[['ds']]
            res.rename(columns={'ds': 'date'}, inplace=True)
            res['predict'] = voting_preds
            # Сохраняем DataFrame в CSV файл
            res.to_csv('predicted_17team.csv', index=False)

        print(f"Предсказания сохранены в файл с именем 'predicted_17team'")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    

if __name__ == "__main__":
    try:
        # Проверяем, передан ли путь к датасету в аргументах командной строки
        if len(sys.argv) != 2:
            raise ValueError("Использование: python main.py <путь_к_датасету>")
        
        # Получаем путь к датасету из аргументов командной строки
        dataset_path = sys.argv[1]
        
        # Попытка обработки датасета
        process_dataset(dataset_path)
    
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def make_features(dataset, column_name, max_lag, rolling_mean_size):
    dataset = dataset.copy()
    dataset['year'] = dataset['ds'].dt.year
    dataset['month'] = dataset['ds'].dt.month
    dataset['day'] = dataset['ds'].dt.day
    dataset['dayofweek'] = dataset['ds'].dt.dayofweek
    dataset['dayofyear'] = dataset['ds'].dt.dayofyear

    for lag in range(1, max_lag + 1):
        dataset['lag_{}'.format(lag)] = dataset[column_name].shift(lag)

    dataset['rolling_mean'] = dataset[column_name].shift().rolling(rolling_mean_size).mean()
    return dataset