# Elephants model exploration

## Hierarchical Token Semantic Audio Transformer (HTS-AT)
Here are the codes for [HTS-AT](https://arxiv.org/abs/2202.00874) based off the [official implementation](https://github.com/RetroCirce/HTS-Audio-Transformer), adapted to our use-case.
Here only the implementation of the model, configuration and training is separate, datasets etc. are the same as in the rest of the repository.

## Files
`utils.py` and `layers.py` come straight from the original repository, `htsat.py` contains modified architecture, slightly adjusted to our use-case with simplifications and deletion of deprecated features to make the code more readable. Same applies to `config.py`, with configuration of the architecture and training. 

`train.py` utilizes our data pipeline to train the model on the ELP data.

## Training
To train the model, run `train.py`. Hyperparameters can be found in `config.py`.

As the HTS-AT uses [Swin Transformer](https://github.com/microsoft/Swin-Transformer) as a part of it's architecture, weights of a pretrained Swin Transformer can be used to speed up the training. 
To get the weights, download checkpoints from the original [repository](https://github.com/microsoft/Swin-Transformer).
By default, "Swin-T/C24" can be used, however changing HTS-AT architecture, will require using according Swin checkpoint (if such exists).
Set `swin_pretrain_path` in `config.py` to load the model.

*Note: It's expected for few parameters to be unfound when loading the pretrained weights.* 

## Requirements
HTS-AT implementation has some additional requirements compared to the rest of our code, thus the additional `requirements.txt` from the original repo.

Additionally *SOX* and *ffmpeg* need to be installed, this can be done in a conda environment with the following:
```
conda install -c conda-forge sox
conda install -c conda-forge ffmpeg
```
*Note: Despite `pytorch_lightning==1.5.9` in requirements.txt, make sure it matches your version of PyTorch. While using PyTorch 2.0, I had to update Lightning to 2.0 too.*
## Model output
Note that the output of the token-semantic module ("framewise_output" of the dictionary returned by the forward method) is of shape `(batch_size, #_of_classes, 1024)`, where 1024 corresponds to the time dimension.
This can be treated as a heatmap of probabilities and that's also why datasets in `train.py` have `event_bins=1024`.
(so 1 timestep ≈ 0,01 s)

The greater time resolution isn't crucial in our case, so the `infer` method of `HTSAT_Swin_Transformer` performs a simple aggregation of timesteps to go from 1024 back to 10, to easily swap between AST and HTS-AT. \
*`infer` is for inference only, during training and in the `forward` method the model still outputs `1024` timesteps.*