from typing import Dict, Union

import torch
import torch.nn as nn

TensorDict = Dict[str, torch.Tensor]


class Processor(nn.Module):
    def forward(self,
                *args: torch.Tensor,
                return_outputs_dict: bool = False,
                **kwargs) -> Union[torch.Tensor, TensorDict]:
        controls = self.get_controls(*args, **kwargs)
        signal = self.get_signal(**controls)

        if return_outputs_dict:
            result = controls.copy()
            result['signal'] = signal
            return result
        else:
            return signal

    def get_controls(self, *args: torch.Tensor,
                     **kwargs: torch.Tensor) -> TensorDict:
        raise NotImplementedError

    def get_signal(self, *args: torch.Tensor,
                   **kwargs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Add(Processor):
    def get_controls(self,
                     signal_one: torch.Tensor,
                     signal_two: torch.Tensor) -> TensorDict:
        return {'signal_one': signal_one, 'signal_two': signal_two}

    def get_signal(self,
                   signal_one: torch.Tensor,
                   signal_two: torch.Tensor) -> torch.Tensor:
        return signal_one + signal_two
