import random
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from knockknock import discord_sender


def negative_sample(df):
    x = df.groupby(['user', 'item'])
    negative_df = pd.DataFrame()
    for name, group in tqdm(x):
        for index, row in tqdm(group.iterrows(), leave=False):
            daytime_options = list(set(columns_unqiue['daytime']) - set(group.daytime.tolist()))
            city_options = list(set(columns_unqiue['city']) - set(group.city.tolist()))
            weather_options = list(set(columns_unqiue['weather']) - set(group.weather.tolist()))
            country_options = list(set(columns_unqiue['country']) - set(group.country.tolist()))
            weekday_options = list(set(columns_unqiue['weekday']) - set(group.weekday.tolist()))
            
            negative_row1 = row.copy()
            negative_row2 = row.copy()

            if daytime_options:
                new_daytime = random.choices(daytime_options, k=2)
                negative_row1.daytime = new_daytime[0]
                negative_row2.daytime = new_daytime[1]


            if city_options:
                new_city = random.choices(city_options, k=2)
                negative_row1.city = new_city[0]
                negative_row2.city = new_city[1]

            if weather_options:
                new_weather = random.choices(weather_options, k=2)
                negative_row1.weather = new_weather[0]
                negative_row2.weather = new_weather[1]

            if country_options:
                new_country = random.choices(country_options, k=2)
                negative_row1.country = new_country[0]
                negative_row2.country = new_country[1]

            if weekday_options:
                new_weekday = random.choices(weekday_options, k=2)
                negative_row1.weekday = new_weekday[0]
                negative_row2.weekday = new_weekday[1]
            
            negative_df = negative_df.append(negative_row1)
            negative_df = negative_df.append(negative_row2)
        
    
    return negative_df

if not os.path.exists('..data/frappe_deepafm_all.csv'):
    
    frappe_df = pd.read_csv('../data/frappe/frappe.csv', sep='\t')
    
    columns_unqiue = {}
    for column in frappe_df.columns:
        columns_unqiue[column] = frappe_df[column].unique().tolist()

    
    negative_df = negative_sample(frappe_df)

    negative_df['label'] = [0] * negative_df.shape[0]

    negative_df = negative_df[['user', 'item', 'cnt', 'daytime', 'weekday', 'isweekend', 'homework', 'cost',
           'weather', 'country', 'city', 'label']]

    frappe_df['label'] = [1] * frappe_df.shape[0]

    df = frappe_df.append(negative_df)

    df.reset_index(drop=True, inplace=True)

    df.to_csv('frappe_deepafm_all.csv', index=False)
