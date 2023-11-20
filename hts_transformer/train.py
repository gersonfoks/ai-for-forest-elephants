import argparse
import os

import pytorch_lightning as pl
import pandas as pd
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.data.dataset import AudioDataset, collate_fn
from src.data.utils import get_base_dir_clips
from src.data.split import get_wav_file_split
from src.metrics import get_metrics

# import htsat
from hts_transformer.htsat import HTSAT_Swin_Transformer
import hts_transformer.config as config

parser = argparse.ArgumentParser(
    description='Creates a single dataframe containing all the information about the sound clips')
parser.add_argument('--dev', action='store_true', help='If we want to use a smaller dev set')
args = parser.parse_args()

batch_size = config.batch_size
max_epochs = config.max_epoch

base_dir = get_base_dir_clips(dev=args.dev)
meta_data_path = os.path.join(base_dir, 'metadata.tsv')
metadata_df = pd.read_csv(meta_data_path, sep='\t',
                          names=['file_reference', 'start_time', 'end_time', 'label'])

train_wav_files, val_wav_files = get_wav_file_split(args.dev, split_percentage=0.8)

train_dataset = AudioDataset('', metadata_df, wav_files=train_wav_files, event_bins=1024)
val_dataset = AudioDataset('', metadata_df, wav_files=val_wav_files, event_bins=1024)

# Here we create a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Get the metrics
train_metrics, val_metrics = get_metrics()

# Initialize the model
model = HTSAT_Swin_Transformer(
    spec_size=config.htsat_spec_size,
    patch_size=config.htsat_patch_size,
    in_chans=1,
    num_classes=config.classes_num,
    window_size=config.htsat_window_size,
    config=config,
    depths=config.htsat_depth,
    embed_dim=config.htsat_dim,
    patch_stride=config.htsat_stride,
    num_heads=config.htsat_num_head,
    train_metrics=train_metrics,
    val_metrics=val_metrics
)

# Checkpointing
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='checkpoints', every_n_epochs=1, save_top_k=-1)

wandb_logger = WandbLogger(project='elephants')
trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger,
                     callbacks=[checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)
