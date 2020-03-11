import os
import neuroanalysis
from neuroanalysis.data.loaders.mies_dataset_loader import MiesNwbLoader
from neuroanalysis.data.dataset import Dataset
from optoanalysis.analyzers import OptoBaselineAnalyzer

opto_file = os.path.join(os.path.dirname(neuroanalysis.__file__), '..', 'test_data', 'nwbs', '2019_06_13_exp1_TH-compressed.nwb')

def test_opto_dataset_loading():

    global opto_file
    dataset = Dataset(loader=MiesNwbLoader(opto_file, baseline_analyzer_class=OptoBaselineAnalyzer))

    assert len(dataset.contents) == 123

    ## test device names -- these might need to change, but we should make sure they change on purpose
    assert set(dataset.contents[0].devices) == set([0, 'LED-590nm', 1])
    assert set(dataset.contents[5].devices) == set([0, 'Prairie_Command', 1, 'Fidelity'])

    ## test data loading / tseries creation
    assert dataset.contents[0]['LED-590nm'].channels == ['reporter']

    assert dataset.contents[0]['LED-590nm'].data().shape == (153500, 1)

    ts = dataset.contents[0]['LED-590nm']['reporter']
    assert ts.time_at(1000) == 0.02
    assert ts.index_at(0.02) == 1000

    ## test PatchClampRecording
    pc_rec = dataset.contents[5][0]
    assert pc_rec.clamp_mode == 'vc'
    assert pc_rec.channels == ['primary', 'command']
    assert pc_rec.device_type == 'MultiClamp 700' ## might change, but should change on purpose




