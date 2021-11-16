from collections import OrderedDict
from pathlib import Path

import torch

from zak.model import Decoder


def load_checkpoint(version: int) -> OrderedDict:
    # file = Path(
    #     Path.cwd(),
    #     'lightning_logs',
    #     'default',
    #     f'version_{version}',
    #     'checkpoints',
    # ).glob('*.ckpt')
    # file = sorted(list(file), key=lambda x: int(x.name.split('-')[0].split('=')[1]))
    # file = file[-1]
    file = 'model.ckpt'

    state_dict = torch.load(file)['state_dict']
    new_state = OrderedDict()
    for key in state_dict.keys():
        if key.startswith('model'):
            new_key = key[6:]
            new_state[new_key] = state_dict[key]

    return new_state


zak = Decoder(live=True, batch_size=1)
zak.load_state_dict(load_checkpoint(2), strict=False)
zak.eval()
for name, val in zak.named_parameters():
    val.requires_grad = False

ts_module = torch.jit.script(zak.cuda())
ts_module.save('zak.pt')
