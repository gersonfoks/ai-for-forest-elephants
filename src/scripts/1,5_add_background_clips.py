# This (optional) step adds pure background clips to the dataframe.
import argparse

import pandas as pd
from src.data.preprocess import add_background_clips
import ast
from src.data.utils import get_training_df_location

parser = argparse.ArgumentParser(
    description='Adds background clips to the dataframe ')

parser.add_argument('--dev',
                    action='store_true', help='If we want to use a smaller dev set')

parser.add_argument('--target_clip_length',
                    type=int, default=20, help='The target clip length in seconds')
args = parser.parse_args()

df_location = get_training_df_location(args.dev)

training_df = pd.read_csv(df_location)

col_with_lists = [
    'Rumble Begin Time (s)',
    'Rumble End Time (s)', 'Rumble File Offset (s)', 'Rumble Duration (s)',
    'Gunshot Begin Time (s)', 'Gunshot End Time (s)',
    'Gunshot File Offset (s)', 'Gunshot Duration (s)',
]


def to_lists(x):
    if type(x) == str:
        return ast.literal_eval(x)
    else:
        return []


for col in col_with_lists:
    training_df[col] = training_df[col].apply(to_lists)

training_df = add_background_clips(training_df)

save_location = get_training_df_location(args.dev)

training_df.to_csv(save_location, index=False)
