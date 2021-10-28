#!/usr/bin/env sh

pip install --upgrade setuptools pip wheel

base=/home/kureta/Documents/other/pytorch-build/wheels
pip install "${base}"/*.whl

pip install pytorch-lightning
pip install JACK-Client librosa matplotlib numpy python-osc SoundFile jupyter
pip install torchcrepe
