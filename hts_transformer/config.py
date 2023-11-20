"""
Code below is based and referred from https://github.com/RetroCirce/HTS-Audio-Transformer
Changes mainly come down to adjusting the code to our SED use-case,
additionally a lot of unused and deprecated was deleted for readability.
"""

batch_size = 32  # batch size per GPU x GPU number , default is 32 x 4 = 128
learning_rate = 1e-3  # 1e-4 also workable 
max_epoch = 1000
num_workers = 3

# I recommend reading this issue to understand this part
# https://github.com/RetroCirce/HTS-Audio-Transformer/issues/18
lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]

# for signal processing
sample_rate = 4000
clip_samples = sample_rate * 10  # 10 for the clip length
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)

# for data collection
classes_num = 2
crop_size = None  # int(clip_samples * 0.5) deprecated

# for htsat hyperparamater
htsat_window_size = 8
htsat_spec_size = 256
htsat_patch_size = 4 
htsat_stride = (4, 4)
htsat_num_head = [4, 8, 16, 32]
htsat_dim = 96 
htsat_depth = [2, 2, 6, 2]

swin_pretrain_path = None 
# "/home/studio-lab-user/model-exploration/hts_transformer/swin_tiny_c24_patch4_window8_256.pth"
# To get the model, download "Swin-T/C24" from https://github.com/microsoft/Swin-Transformer