import os
import h5py
from neuroanalysis.data.loaders.mies_dataset_loader import MiesNwbLoader
from neuroanalysis.data.loaders.acq4_dataset_loader import Acq4DatasetLoader
from neuroanalysis.data.dataset import Dataset
from optoanalysis.analyzers import OptoBaselineAnalyzer
from aisynphys.analyzers import MPBaselineAnalyzer
from neuroanalysis.miesnwb import MiesNwb
import pyqtgraph as pg

pg.dbg()


f = "/Users/meganbkratz/Code/ai/example_data/data/2019-06-13_000/slice_000/site_000/2019_06_13_exp1_TH-compressed.nwb"
f2 = "/Users/meganbkratz/Code/ai/example_data/2019_06_24_131623-compressed.nwb"
f3 = "/Users/meganbkratz/Documents/ManisLab/L4Mapping/ExcitationProfileData/2012.11.09_000/slice_000/cell_004"

#hdf = h5py.File(f, 'r')

mies_nwb = Dataset(loader=MiesNwbLoader(f2, baseline_analyzer_class=MPBaselineAnalyzer))
mies_nwb_old = MiesNwb(f2)
opto_nwb = Dataset(loader=MiesNwbLoader(f, baseline_analyzer_class=OptoBaselineAnalyzer))
acq4_dataset = Dataset(loader=Acq4DatasetLoader(f3))

#old = OptoNwb(f)


### for profiling lazy load stimulus vs stimulus
# prof = pg.debug.Profiler(disabled=False)

# for srec in mies_nwb.contents:
#     recs = srec.recordings

# prof('made recordings')

# for srec in mies_nwb.contents:
#     for rec in srec.recordings:
#         desc = rec.stimulus.description

# prof('got stimulus descriptions')
# prof.finish()

