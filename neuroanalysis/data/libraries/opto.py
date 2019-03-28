import json
import math
import os.path
from collections import OrderedDict
from neuroanalysis.data.cell import Cell
from neuroanalysis.data.electrode import Electrode

def get_cells(expt):
    """Return a dictionary of {cell_id:Cell(), ...} for all cells in experiment."""

def get_region(expt):
    """Return the region of the brain that the experiment took place in, ie. V1, A1, etc."""

def get_site_info(expt):
    """Return a dict with info from the .index file for the site of this experiment.
        Must include:
            __timestamp__   The timestamp of when this experiment started."""
    ### TODO: define what is expected here for non-acq4 users

def get_slice_info(expt):
    """Return a dict with info from the .index file for this experiment's slice."""
    ### TODO: define what is expected here for non-acq4 users

#def get_slice_directory(expt):
#    """Return a path to the directory of the slice."""
#    ### This makes an assumption that there is a directory for the slice. This seems like an acq4 specific assumption

def get_expt_info(expt):
    """Return a dict with info from the .index file for this experiment."""
    ### TODO: define what is expected here for non-acq4 users

def load_from_file(expt, file_path):
    """First function that is called during expt initialization. Load information from the file at file_path
    to populate fields in expt. Must populate: electrodes, _cells

    For opto, file_path leads to a connections.json file. A file from either acq4's mosaic editor or new_test_ui.py is accepted.
    """
    expt.electrodes = OrderedDict()
    expt._cells = OrderedDict()

    filename = os.path.basename(file_path)
    expt._uid=filename[0:-17]
    expt.source_id = (filename, None)

    with open(filename,'r') as f:
        exp_json=json.load(f)
    version = exp_json.get('version', None)

    if version is None:
        load_markPoints_file(expt, exp_json)
    else:
        load_connection_file(expt, exp_json)


def process_meta_info(expt, meta_info):
    """Process optional meta_info dict that is passed in at the initialization of expt.
    Called after load_from_file."""

def get_uid(expt):
    return expt._uid

##### Private functions: ######

def load_markPoints_file(expt, exp_json):

    ## load stim point positions
    tseries_keys=[key for key in exp_json.keys() if 'TSeries' in key]
    points = OrderedDict()
    for tseries in tseries_keys:
        for point, data in exp_json[tseries]['MarkPoints'].items():
            pos = (data['x_pos']*1e-6, data['y_pos']*1e-6, data['z_pos']*1e-6)
            points[point] = {'pos':pos}


    ##load electrode positions and connections
    HS_keys=[key for key in exp_json['Headstages'].keys() if key.startswith('electrode')]
    for headstage in HS_keys:
        data = exp_json['Headstages'][headstage]
        elec = Electrode(headstage, start_time=None, stop_time=None, device_id=headstage[-1])
        expt.electrodes[headstage] = elec
        cell = Cell(expt, headstage, elec)
        cell.position = (data['x_pos']*1e-6, data['y_pos']*1e-6, data['z_pos']*1e-6) #covert from um to m
        cell.angle = data['angle']
        cell.has_readout = True
        cell.has_stimulation = True ## it's possible to interogate patched cell pairs, even if that's not employed often
        expt._cells[cell.cell_id]=cell
        for p, conn in data['Connections'].items():
            if conn is None: ## skip this one, it doesn't have a match in points and is a duplicate
                continue
            points[p][headstage] = conn



    ## check points for overlappingness (same as MP_trim in old analysis)
    distance = 10e-6
    skip=[]
    for i, p1 in enumerate(points.keys()):
        for p2 in points.keys()[i+1:len(points)]:
            x_dif = points[p2]['pos'][0] - points[p1]['pos'][0]
            y_dif = points[p2]['pos'][1] - points[p1]['pos'][1]
            z_dif = points[p2]['pos'][2] - points[p1]['pos'][2]
            xy_dif=math.sqrt(x_dif**2+y_dif**2)
            xyz_dif=math.sqrt(x_dif**2+y_dif**2+z_dif**2)
            if xyz_dif < distance: 
                same = True
                for hs in expt.electrodes.keys():
                    if points[p1][hs] != points[p2][hs]:
                        same=False
                if not same:
                    skip.append(p1)

    ## create cells for points that were not overlapping
    for point, data in points.items():
        if point not in skip:
            cell = Cell(expt, point, None)
            cell.position = data['pos']
            cell.has_readout = False
            cell.has_stimulation = True
            expt._cells[cell.cell_id]=cell

    ### create Pairs for all tested connections
    # for point in points:
    #     for hs in HS_keys:
    #         expt.pairs[(point, hs)]._connection_call = point[hs]
    for p in expt.pairs.values():
        if p.preCell.cell_id in points:
            if p.postCell.cell_id in HS_keys:
                p._connection_call = points[p.preCell.cell_id][p.postCell.cell_id]




            
    


