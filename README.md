# Elephants model exploration

## Getting started

Install the requirements from the requirements.txt file. I would recommend using python 3.10

Furthermore, install pytorch and torchaudio.

Create a config file `config/connection_config.yaml` containing the following information:

```
service_name: 's3'
region_name: 'us-east-2'
aws_access_key_id: <to fill in>
aws_secret_access_key: <to fill in>
```

## Getting the clips

In the src.scripts there are 4 scripts that you can run to get clips from the available data.
At the end you have the clips available that are at most n (standard=20) seconds long. There are 2 classes:

- Rumble
- Gunshots


Run scripts 1 up to 4 one by one.

```
python -m src.scripts.1_create_training_df 
```
`1,5_add_background_clips.py` will add clips with neither gunshots nor rumbles to the training dataset, theoretically this is an optional step. We've noticed only a ~1% drop in accuracy on the validation set when training with these clips added, while this should greatly reduce the amount of false positives on real data.
If you don't want the background clips, besides not running this script, make sure to delete `Background` from  `LABELS` in `src.data.utils`

Optionally you can use the `--dev` flag in every step for generating a smaller set for development.

These scripts will download the files to a `./data` directory.

### Clip length

As mentioned, the default length of the clips that we create is 20 seconds, while the models take in 10 second clips. That's because, in `__data_generation` of the `AudioDataset` we sample a 10 second window out of these 20 second clips. This has two purposes:
1. Randomizes the position of events in the clip.
2. Serves as an data augmentation method.

### Labels
By default, the labels returned by `AudioDataset` are of `(batch_size, 2, 10)` shape, where 2 is the number of classes and 10 are the timesteps (so timestep is equal to 1 second with 10 second clips). 
These are controlled by `LABELS` in `src.data.utils` (excluding `Background`, which isn't really a label) and `event_bins` parameter of `AudioDataset` respectively. 
`event_bins` is useful to adapt the dataset to other clip lengths or models such as the HTS-AT which always outputs 1024 timesteps.
For each timestep, 1 in a row of class X, means that that sound is present in it, 0 means that it's not - in some sense this can be viewed as a binary classification.

## Training

The training script is under `./src/scripts/training/train_model`.
It accepts a config file as an input and will train the model based on the config file.

## Project structure

The project structure is as follows:

```
config: contains the config for connection to aws
data: place where all the data is stored as well as a dataset class for iterating of the data
hts_transformers: contains the hts_transformers library
notebooks: contains notebooks for exploration
src: contains all the source code
    data: contains the data pipeline
    models: contains code for the models. Note that we use an abstract model for common functionality and inherit from that
    scripts: contains scripts for data processing and training
tests: contains tests for the code
```

##  Tips on further development

- Use the `--dev` flag in the scripts to generate a smaller dataset for development
- Use the abstract model in `src.models.abstract_model` for common functionality and easy of development.

You can always contact the team for more information.