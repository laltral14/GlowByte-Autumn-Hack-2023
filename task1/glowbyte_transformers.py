# glowbyte_transformers.py

import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


def check_percentage_condition(string):
    # Находим все числа в строке
    numbers = [int(s.strip('%,')) for s in string.split() if s.rstrip('%,').isdigit()]

    # Проверяем, есть ли числа в строке
    if numbers:
        # Находим проценты в строке
        percentages = [num for num in numbers if num >= 0 and num <= 100]

        # Проверяем, есть ли проценты в строке
        if percentages:
            # Если есть хотя бы один процент больше 50, возвращаем True
            if any(percentage > 50 for percentage in percentages):
                return True

    # Если условие не выполняется, возвращаем False
    return False


def generalize_condition(row):
    condition = row['weather_pred']
    if pd.isna(condition) or isinstance(condition, (int, float)):
        return 'неизвестно'  # Обработка NaN и значений типа float
    elif 'ясно' in condition:
        return 'ясно'
    elif 'дожд' in condition or 'ливен' in condition or 'гроз' in condition or 'град' in condition or 'шторм' in condition or 'гром' in condition or 'ливн' in condition or 'дожь' in condition:
        return 'дождь/ливень/гроза'
    elif 'снег' in condition or 'снеж' in condition or 'метель' in condition or 'Снег' in condition or 'нег' in condition or 'д+сн' in condition:  
        return 'снег'
    elif 'туман' in condition or 'дымка' in condition or 'морось' in condition or 'моромь' in condition or 'моровь' in condition or 'изморозь' in condition or 'заморозки' in condition:
        return 'дождь/ливень/гроза'
    elif 'малообл' in condition or 'ясн' in condition or 'солнечно' in condition or 'малооб' in condition:
        return 'ясно'
    elif 'пасм' in condition or 'обл' in condition:
        return 'пасмурно'
    elif check_percentage_condition(condition):
        if row['temp_pred'] > 2:
            return 'дождь'
        else:
            return 'снег'
    else:
        return 'ясно'


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = pd.DataFrame()
        X_transformed['datetime'] = pd.to_datetime(X['date'] + '-' + X['time'].astype(str), format='%Y-%m-%d-%H')
        X_transformed[['target', 'temp_pred', 'weather_pred']] = X[['target', 'temp_pred', 'weather_pred']]
        return X_transformed
    

class GeneralizeConditionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['weather'] = X.apply(generalize_condition, axis=1)
        return X[['datetime', 'target', 'temp_pred', 'weather']]
    

class FillNATransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        columns_to_fillna = ['temp_pred', 'weather_pred']
        X[columns_to_fillna] = X[columns_to_fillna].fillna(method='ffill')
        return X


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, column_mapping):
        self.columns = columns
        self.column_mapping = column_mapping
        self.ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore', categories='auto')

    def fit(self, X, y=None):
        self.ohe.fit(X[self.columns])
        return self

    def transform(self, X):
        temp_df = pd.DataFrame(data=self.ohe.transform(X[self.columns]), columns=self.ohe.get_feature_names_out(self.columns))
        X_ohe = pd.concat([X.reset_index(drop=True), temp_df], axis=1)
        X_ohe = X_ohe.drop(columns=self.columns, axis=1)
        
        # Переименование столбцов с использованием column_mapping
        X_ohe = X_ohe.rename(columns=self.column_mapping)
        return X_ohe


class ResampleDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.set_index('datetime', inplace=True)
        resampled_df = X.resample('D').agg({
            'target': 'sum',
            'temp_pred': ['mean', 'min', 'max'],
            'rain': 'max',
            'cloudy': 'max',
            'snow': 'max',
            'clear': 'max'
        })

        resampled_df.columns = ['target_sum', 'temp_pred_mean', 'temp_pred_min', 'temp_pred_max', 'rain', 'cloudy', 'snow', 'clear']
        return resampled_df
    

class HolidaysTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, data):
        def create_holidays_dataset(data):
            start_date = data.index.min()
            end_date = data.index.max()

            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            years = data.index.year.unique()

            holidays_dict = {date: 'yes' for date in holidays.RU(years=years) if start_date <= date <= end_date}

            df_holidays = pd.DataFrame(list(holidays_dict.items()), columns=['ds', 'holiday'])
            df_holidays['ds'] = pd.to_datetime(df_holidays['ds'])
            return df_holidays

        df_holidays_train = create_holidays_dataset(data)

        data = data.rename(columns={'target_sum':'target'}).rename_axis('ds')
        data = pd.merge(data, df_holidays_train, on='ds', how='left')
        data['holiday'] = data['ds'].isin(df_holidays_train['ds']).apply(lambda x: 1 if x else 0)

        return data
    

class CovidTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.date_ranges = [
            ('2020-04-28', '2020-05-31'),
            ('2020-10-19', '2021-01-22'),
            ('2021-06-02', '2021-06-20'),
            ('2021-10-28', '2021-11-07')
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        def create_covid_dataset(date_ranges):
            data_covid = {'ds': [], 'covid': []}

            for start_date, end_date in date_ranges:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')

                current_date = start_date
                while current_date <= end_date:
                    data_covid['ds'].append(current_date.strftime('%Y-%m-%d'))
                    data_covid['covid'].append(1)
                    current_date += timedelta(days=1)

            df_covid = pd.DataFrame(data_covid)
            df_covid['ds'] = pd.to_datetime(df_covid['ds'])
            return df_covid

        df_covid_train = create_covid_dataset(self.date_ranges)

        data['ds'] = pd.to_datetime(data['ds'])
        data = pd.merge(data, df_covid_train, on='ds', how='left')
        data['covid'] = data['ds'].isin(df_covid_train['ds']).apply(lambda x: 1 if x else 0)

        return data