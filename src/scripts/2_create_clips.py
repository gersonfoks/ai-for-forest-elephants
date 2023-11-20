# This step creates the actual clips and put them in a datafile.
import argparse

import pandas as pd
from src.data.preprocess import get_clips_from_file
from tqdm import tqdm
import ast
import os
from scipy.io import wavfile
from src.data.utils import get_s3_client_and_resource, get_base_dir_clips, \
    ClipManager, LABELS, create_dirs_for_file, get_training_df_location

parser = argparse.ArgumentParser(
    description='Creates the clips and saves them ')

parser.add_argument('--dev',
                    action='store_true', help='If we want to use a smaller dev set')
args = parser.parse_args()

client, s3_resource = get_s3_client_and_resource()

df_location = get_training_df_location(args.dev)

training_df = pd.read_csv(df_location)

col_with_lists = []
for sound_class in LABELS:
    col_with_lists.extend([f'{sound_class} Begin Time (s)',
                           f'{sound_class} End Time (s)',
                           f'{sound_class} File Offset (s)',
                           f'{sound_class} Duration (s)'])


def to_lists(x):
    if type(x) == str:
        return ast.literal_eval(x)
    else:
        return []


for col in col_with_lists:
    training_df[col] = training_df[col].apply(to_lists)

# Get the basedir:
base_dir = get_base_dir_clips(args.dev)
clip_manager = ClipManager(base_location=base_dir)

metadata_path = os.path.join(base_dir, 'metadata.tsv')
create_dirs_for_file(metadata_path)

with open(metadata_path, 'w') as f:
    for i, row in tqdm(training_df.iterrows(), total=len(training_df.index)):
        clips = get_clips_from_file(s3_resource, row, LABELS)
        for j, clip in enumerate(clips):
            loc_dir, save_location = clip_manager.get_loc(clip)
            os.makedirs(loc_dir, exist_ok=True, )
            wavfile.write(save_location, clip.sample_rate, clip.data)

            # Save events within the clip to the metadata file:
            events = clip.events
            for event in events:
                if clip.label != 'Background':
                    f.write(f'{save_location}\t{event[0]}\t{event[1]}\t{clip.label}\n')
                else:
                    f.write(f'{save_location}\t\t\t\n')