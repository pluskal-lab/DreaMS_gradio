#!/bin/bash

# Install DreaMS (in editable mode for debugging)
git clone https://github.com/pluskal-lab/DreaMS.git
cd DreaMS
pip install -e .

# Download spectra library
wget -P ./DreaMS/data https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary/MassSpecGym_DreaMS.hdf5

cd ..
