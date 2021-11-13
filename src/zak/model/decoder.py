import torch.nn as nn
from torch import Tensor

from zak.ddsp import Noise, OscillatorBank, Reverb
from zak.model.controller import Controller


class Decoder(nn.Module):
    def __init__(self,
                 batch_size: int = 6,
                 sample_rate: int = 48000,
                 hop_size: int = 480,
                 n_harmonics: int = 126,
                 n_noise_filters: int = 150,
                 reverb_duration: float = 1.0,
                 decoder_mlp_units: int = 1024,
                 decoder_mlp_layers: int = 3,
                 decoder_gru_units: int = 1024,
                 decoder_gru_layers: int = 1,
                 live: bool = False):
        super().__init__()
        self.live = False
        self.controller = Controller(
            n_harmonics,
            n_noise_filters,
            decoder_mlp_units,
            decoder_mlp_layers,
            decoder_gru_units,
            decoder_gru_layers,
            live
        )
        self.harmonics = OscillatorBank(
            batch_size,
            sample_rate,
            n_harmonics,
            hop_size,
            live
        )
        self.noise = Noise(
            sample_rate,
            hop_size,
            batch_size,
            live
        )
        self.reverb = Reverb(
            sample_rate,
            reverb_duration,
            batch_size,
            live
        )

    def forward(self, f0: Tensor, loudness: Tensor) -> Tensor:
        harmonics_ctrl, noise_ctrl = self.controller(f0, loudness)
        harmonics = self.harmonics(*harmonics_ctrl)
        noise = self.noise(noise_ctrl)

        signal = harmonics + noise
        signal += self.reverb(signal)

        return signal
