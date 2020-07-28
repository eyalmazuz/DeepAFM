import random
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from knockknock import discord_sender


def negative_sample(df, columns_unqiue):
    x = df.groupby(['userId', 'movieId'])
    negative_df = pd.DataFrame()
    for name, group in tqdm(x):
        for index, row in tqdm(group.iterrows(), leave=False):
            tag_options = list(set(columns_unqiue['tag']) - set(group.tag.tolist()))
           

            negative_row1 = row.copy()
            negative_row2 = row.copy()

            if tag_options:
                new_tag = random.choices(tag_options, k=2)
                negative_row1.tag = new_tag[0]
                negative_row2.tag = new_tag[1]
    
            negative_df = negative_df.append(negative_row1)
            negative_df = negative_df.append(negative_row2)
        
    
    return negative_df

if not os.path.exists('../data/movielens_all.csv'):
    
    movielens_df = pd.read_csv('../data/ml-20m/tags.csv', sep=',')
    
    columns_unqiue = {}
    for column in movielens_df.columns:
        columns_unqiue[column] = movielens_df[column].unique().tolist()

    
    negative_df = negative_sample(movielens_df, columns_unqiue)

    negative_df['label'] = [0] * negative_df.shape[0]

    negative_df = negative_df[['userId', 'movieId', 'tag']]

    movielens_df['label'] = [1] * movielens_df.shape[0]

    df = movielens_df.append(negative_df)

    df.reset_index(drop=True, inplace=True)

    df.to_csv('movielens_all.csv', index=False)
