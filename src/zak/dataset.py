from pathlib import Path

import torch
import torch.nn.functional as F  # noqa
from torch.utils.data import Dataset

saved_dir = Path('~/Music/test/').expanduser()


def load_saved(path: Path, name: str):
    return torch.load(path.joinpath(name))


def get_examples(data: torch.Tensor, duration: int, hop: int):
    return data.unfold(1, duration, hop).permute(1, 0, 2)


class Parameters(Dataset):
    def __init__(self):
        audio = load_saved(saved_dir, 'audio.pth')
        pitch = load_saved(saved_dir, 'pitch.pth')
        loudness = load_saved(saved_dir, 'loudness.pth')

        # 480 audio samples = 1 feature step
        # Get 4 second examples with 1 second overlap
        self.audio = get_examples(audio, 2*48000, 1*48000)
        self.pitch = get_examples(pitch, 2*100, 1*100)
        self.loudness = get_examples(loudness, 2*100, 1*100)

    def __len__(self):
        return self.audio.shape[0]

    def __getitem__(self, item):
        return self.audio[item], self.pitch[item], self.loudness[item]
