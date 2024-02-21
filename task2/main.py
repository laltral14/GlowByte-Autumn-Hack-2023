import sys
import pickle
import pandas as pd
import numpy as np
from glowbyte_transformers import DatetimeTransformer, FillNATransformer, GeneralizeConditionTransformer, OneHotEncoderTransformer, ResampleDataTransformer, HolidaysTransformer, CovidTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def process_dataset(dataset_path):
    # Здесь вы можете добавить код для обработки датасета
    print(f"Запуск прогнозирования для датасета по пути: {dataset_path}")
    
    try:
        # Загрузка модели из файла с использованием pickle
        with open('voting_model_imp.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            print("Модель успешно загружена:", loaded_model)
            train_df = pd.read_csv('train_test_df.csv')
            full_df = pd.read_csv(dataset_path)
            #Rename columns
            column_mapping = {'weather_пасмурно': 'cloudy', 'weather_снег': 'snow', 'weather_ясно':
                              'clear', 'weather_дождь/ливень/гроза': 'rain', 'weather_туман': 'fog'}
            #Pipeline steps
            pipeline = Pipeline(steps=[
                ('datetime_transform', DatetimeTransformer()),
                ('fill_na', FillNATransformer()),
                ('generalize_condition', GeneralizeConditionTransformer()),
                ('one_hot_encode', OneHotEncoderTransformer(columns=['weather'], column_mapping=column_mapping)),
                ('holidays', HolidaysTransformer()),
                ('covid', CovidTransformer()),
                #('working_hour', WorkingHoursTransformer()),
            ])
            
            lag_list = list(range(24, 49))
            lag_list.append(72)
            lag_list.append(96)
            lag_list.append(120)
            lag_list.append(144)

            def make_features(dataset, column_name, lag_list):
                
                dataset = dataset.copy()
                dataset['year'] = dataset['ds'].dt.year
                dataset['month'] = dataset['ds'].dt.month
                dataset['day'] = dataset['ds'].dt.day
                dataset['dayofweek'] = dataset['ds'].dt.dayofweek
                dataset['dayofyear'] = dataset['ds'].dt.dayofyear

                for lag in lag_list:
                    dataset['lag_{}'.format(lag)] = dataset[column_name].shift(lag)

                dataset['diff_lag'] = dataset[column_name].shift(24) - dataset[column_name].shift(48)

                dataset['min_prev_day'] = dataset[column_name].shift(24).rolling(window=24).min()
                dataset['max_prev_day'] = dataset[column_name].shift(24).rolling(window=24).max()

                return dataset


            pipeline.fit(train_df)            
            full_transformed = pipeline.transform(full_df)

            data_f = make_features(full_transformed, 'target', lag_list)

            # Определяем индекс, по которому разделить данные
            split_index = len(train_df)

            # Разделяем данные
            test_transformed = data_f.iloc[split_index:, :]
            test_transformed_back = data_f.iloc[split_index:, :]
            
            test_transformed_back.set_index('ds', inplace=True)

            X_test = test_transformed_back.drop(['target'], axis=1)
            y_test = test_transformed_back['target']

            voting_preds = loaded_model.predict(X_test)

            print(f'MAE: {mean_absolute_error(y_test, voting_preds)}')
            print(f'RMSE: {mean_squared_error(y_test, voting_preds, squared=False)}')
            print(f'MAPE: {mean_absolute_percentage_error(y_test, voting_preds)}')
            print(f'R2: {r2_score(y_test, voting_preds)}')

            res = test_transformed.copy()
            res = res[['ds']]
            res.rename(columns={'ds': 'datetime'}, inplace=True)
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
