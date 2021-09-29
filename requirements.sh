#!/usr/bin/env sh

pip install --upgrade "$(pip list -o --format freeze | cut -d '=' -f 1)"
pip install wheel

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning
pip install JACK-Client librosa matplotlib numpy python-osc SoundFile jupyter
pip install torchcrepe
