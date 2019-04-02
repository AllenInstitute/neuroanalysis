import json
import math
import os.path
from collections import OrderedDict
from neuroanalysis.data.cell import Cell
from neuroanalysis.data.electrode import Electrode
import pyqtgraph as pg

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
    if expt._site_path is not None:
        index = os.path.join(self.expt_path, '.index')
        if not os.path.isfile(index):
            raise TypeError("Cannot find index file (%s) for experiment %s" % (index, self))
        info = pg.configfile.readConfigFile(index)['.']
        info.update(expt._expt_info)
        expt._expt_info = info
    else:
        expt._expt_info = {}
    return expt._expt_info

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
        load_markPoints_connection_file(expt, exp_json)
    else:
        load_mosaiceditor_connection_file(expt, exp_json)


def process_meta_info(expt, meta_info):
    """Process optional meta_info dict that is passed in at the initialization of expt.
    Called after load_from_file."""
    ## Need to load: presynapticCre, presynapticEffector, [class, reporter, layer for each headstage], internal

    preEffector = meta_info['presynapticEffector']
    for e_id, elec in expt.electrodes.items():
        n = e_id[-1]
        cell = elec.cell
        cell._morphology['initial_call'] = meta_info['HS%s_class'%n]
        cell._target_layer = meta_info['HS%s_layer'%n]
        if meta_info['HS%s_reporter'%n] == 'positive':
            cell._cre_type = meta_info['presynapticCre']
        label_cell(cell, preEffector, meta_info['HS%s_reporter'%n] == 'positive')
        elec._internal_solution = meta_info['internal']
        if len(meta_info['distances']) > 0: ### we may not have distance measurements for all cells
            dist = [e for e in meta_info['distances'] if e['headstage']==n]
            if len(dist) > 1:
                raise Exception('Something is wrong.')
            cell.distance_to_pia = dist[0]['toPia']*1e-6
            cell.distance_to_WM = dist[0]['toWM']*1e-6


    for i, cell in expt.cells.items():
        if not cell.has_readout and cell.has_stimulation:
            cell._cre_type = meta_info['presynapticCre']
            label_cell(cell, preEffector, positive=True) ## assume all non-patched stimulated cells are positive for preEffector

    expt.expt_info
    expt.expt_info['internal_solution'] = meta_info['internal'] 



def get_uid(expt):
    return expt._uid

##### Private functions: ######

def label_cell(cell, preEffector, positive=True):
    """Populate appropriate labels for a cell positive for the preEffector."""

    ## Wanted to separate this function out so it is easier to change/amend the logic later

    ## if the cell is positive for the preEffector, populate genetic labels
    if positive:
        cell.labels[preEffector] = True
        if preEffector == 'Ai167':
            cell.labels['tdTomato'] = True
        elif preEffector == 'Ai136':
            cell.labels['EYFP'] = True

    ## if the cell is patched (has an electrode), populate dyes 
    ##    -- this assumes which dye is used for the whole experiment based on the color of the preEffector
    if cell.electrode is not None:
        if preEffector == 'Ai167':
            cell.labels['AF488'] = True
        elif preEffector == 'Ai136':
            cell.labels['AF594'] = True


def load_markPoints_connection_file(expt, exp_json):

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
        elec.cell = cell
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
                if same:
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


def load_mosaiceditor_connection_file(expt, exp_json):
    ## create Cells for stimulation points
    for name, data in exp_json['StimulationPoints'].items():
        if data['onCell']:
            cell = Cell(expt, name, None)
            cell.position = tuple(data['position'])
            cell.has_readout = False
            cell.has_stimulation = True
            expt._cells[cell.cell_id] = cell

    ## create Cells for recorded cells
    for name, data in exp_json['Headstages'].items():
        elec = Electrode(name, start_time=None, stop_time=None, device_id=name[-1])
        cell = Cell(expt, name, elec)
        elec.cell = cell
        cell.position = (data['x_pos'], data['y_pos'], data['z_pos'])
        cell.angle = data['angle']
        cell.has_readout = True
        cell.has_stimulation = True
        expt._cells[cell.cell_id] = cell

    ## populate pair values
    for p in expt.pairs.values():
        try:
            p.connection_call = exp_json['Headstages'][p.postCell.cell_id]['Connections'][p.preCell.cell_id]
        except KeyError:
            print("Could not find connection call for Pair %s -> %s in experiment %s" % (p.preCell.cell_id, p.postCell.cell_id, expt.uid))



            
    


