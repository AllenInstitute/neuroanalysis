import os
import h5py
from neuroanalysis.data.loaders import MiesNwbLoader
from neuroanalysis.data.dataset import Dataset


f = "/Users/meganbkratz/Code/ai/example_data/data/2019-06-13_000/slice_000/site_000/2019_06_13_exp1_TH-compressed.nwb"
f2 = "/Users/meganbkratz/Code/ai/example_data/2019_06_24_131623-compressed.nwb"

#hdf = h5py.File(f, 'r')

mies_nwb = Dataset(loader_class=MiesNwbLoader, file_path=f2)
opto_nwb = Dataset(loader_class=MiesNwbLoader, file_path=f)

#old = OptoNwb(f)