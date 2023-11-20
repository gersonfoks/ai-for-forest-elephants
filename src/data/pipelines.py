import torch
from torchaudio.transforms import Spectrogram, MelScale


class MelSpectogramPipeline(torch.nn.Module):
    def __init__(self, sample_rate=4000, n_fft=1024, n_mels=128, pad_to=80, pad=True, do_normalize=True):
        super(MelSpectogramPipeline, self).__init__()
        self.spec = Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale = MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1
        )

        self.pad_to = pad_to
        self.pad = pad

        self.do_normalize = do_normalize

        # Taken from: https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTFeatureExtractor
        # Might use other values depending on the dataset
        self.mean = 15
        self.std = 1.5

    def forward(self, data):
        specs = self.spec(data)
        mels = torch.log(self.mel_scale(specs))

        if self.do_normalize:
            mels = self.normalize(mels)

        # Pad to 80
        # Pad zeros to the end of the sequence
        if self.pad and mels.shape[2] < self.pad_to:
            zero_padding = torch.zeros(
                (mels.shape[0], mels.shape[1], self.pad_to - mels.shape[2])).to(
                mels.device)
            mels = torch.cat((mels, zero_padding), dim=2)

        return mels

    def normalize(self, mel_scale):
        return (mel_scale - self.mean) / (self.std * 2)
