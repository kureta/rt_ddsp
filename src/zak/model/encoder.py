from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torchcrepe
import torch
import librosa
import tqdm

SAMPLES_PATH = Path('~/Music/violin/Violin Samples').expanduser()
SAMPLE_RATE = 48000
HOP_SIZE = 480


# TODO: Default parameters are:
#       sample_rate = 48000
#       window_size = 3072
#       hop_size = 480
#       frequency range for violin is ~ 190Hz - 2800Hz
def encode(audio: torch.Tensor,
           sample_rate: int = 48000,
           hop_size: int = 480,
           f_min: float = 190.0,
           f_max: float = 2800.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # Select a model capacity--one of "tiny" or "full"
    model = 'full'

    # Choose a device to use for inference
    device = 'cuda:0'

    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 2048

    # Compute pitch using first gpu
    pitch, periodicity = torchcrepe.predict(audio,
                                            sample_rate,
                                            hop_size,
                                            f_min,
                                            f_max,
                                            model,
                                            batch_size=batch_size,
                                            device=device,
                                            decoder=torchcrepe.decode.weighted_argmax,
                                            return_periodicity=True)

    # Filter silence
    periodicity = torchcrepe.threshold.Silence(-90.)(periodicity,
                                                     audio,
                                                     sample_rate,
                                                     hop_size)

    # calculate loudness
    loudness = torchcrepe.loudness.a_weighted(audio, sample_rate, hop_size)

    return pitch, loudness, periodicity


def load_and_resample(file: Union[str, Path]) -> np.ndarray:
    audio, _ = librosa.load(file, SAMPLE_RATE)

    return audio


def main() -> None:
    audio_path = SAMPLES_PATH.parent.joinpath('audio.pth')
    if audio_path.exists():
        audio = torch.load(audio_path)
        print('Audio loaded.')
    else:
        files = sorted(SAMPLES_PATH.glob('*.wav'))
        with Pool(8) as p:
            audio = np.concatenate(p.map(load_and_resample, tqdm.tqdm(files)))
            print('Resampled.')
        audio = torch.from_numpy(audio.astype('float32')).unsqueeze(0)

        print('Torched.')
        torch.save(audio, SAMPLES_PATH.parent.joinpath('audio.pth'))
        print('Audio saved.')

    pitch, loudness, periodicity = [], [], []
    for idx in tqdm.trange(0, audio.shape[1], SAMPLE_RATE * 10):
        p, l, h = encode(audio[:, idx:idx + SAMPLE_RATE * 10])
        pitch.append(p[:, :-1])
        loudness.append(l[:, :-1])
        periodicity.append(h[:, :-1])
    print('Encoded.')

    pitch = torch.cat(pitch, dim=1)
    loudness = torch.cat(loudness, dim=1)
    periodicity = torch.cat(periodicity, dim=1)

    torch.save(pitch, SAMPLES_PATH.parent.joinpath('pitch.pth'))
    torch.save(loudness, SAMPLES_PATH.parent.joinpath('loudness.pth'))
    torch.save(periodicity, SAMPLES_PATH.parent.joinpath('periodicity.pth'))
    print('Saved.')


if __name__ == '__main__':
    main()
