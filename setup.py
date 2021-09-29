#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "wheel",
    "torch==1.9.1+cu111",
    "torchvision==0.10.1+cu111",
    "torchaudio==0.9.1",
    "pytorch-lightning",
    "JACK-Client",
    "librosa",
    "numpy",
    "scipy",
    "python-osc",
    "SoundFile",
    "torchcrepe",
]

test_requirements = ['pytest>=3', ]

setup(
    author="Sahin Kureta",
    author_email='skureta@gmail.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    description="Realtime DDSP implementation in PyTorch",
    entry_points={
        'console_scripts': [
            'rt_ddsp=rt_ddsp.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rt_ddsp',
    name='rt_ddsp',
    packages=find_packages(include=['rt_ddsp', 'rt_ddsp.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/kureta/rt_ddsp',
    version='0.1.0',
    zip_safe=False,
)
