import json
import math, glob
import os
from collections import OrderedDict
from neuroanalysis.data.cell import Cell
from neuroanalysis.data.electrode import Electrode
import pyqtgraph as pg
from multipatch_analysis.util import timestamp_to_datetime
from multipatch_analysis import config

#def get_cells(expt):
#    """Return a dictionary of {cell_id:Cell(), ...} for all cells in experiment."""

#def get_region(expt):
#    """Return the region of the brain that the experiment took place in, ie. V1, A1, etc."""
def path(expt):
    return os.path.join(config.synphys_data, expt._site_path)

def get_site_info(expt):
    """Return a dict with info from the .index file for the site of this experiment.
        Must include:
            __timestamp__   The timestamp of when this experiment started."""
    index = os.path.join(expt.path, '.index')
    if not os.path.isfile(index):
        return None
    return pg.configfile.readConfigFile(index)['.']
    ### TODO: define what is expected here for non-acq4 users

def get_slice_info(expt):
    """Return a dict with info from the .index file for this experiment's slice."""
    index = os.path.join(os.path.split(expt.path)[0], '.index')
    if not os.path.isfile(index):
        return {}
    return pg.configfile.readConfigFile(index)['.']


def get_slice_directory(expt):
    """Return a path to the directory of the slice."""
    ### This makes an assumption that there is a directory for the slice. This seems like an acq4 specific assumption
    return os.path.split(expt.path)[0]

def get_expt_info(expt):
    """Return a dict with info from the .index file for this experiment."""
    if expt._expt_info is None:
        expt_path = os.path.split(os.path.split(expt.path)[0])[0]
        index = os.path.join(expt_path, '.index')
        if not os.path.isfile(index):
            #raise TypeError("Cannot find index file (%s) for experiment %s" % (index, expt))
            expt._expt_info = {}
            return expt._expt_info
        info = pg.configfile.readConfigFile(index)['.']
        expt._expt_info = info
    return expt._expt_info

def last_modification_time(expt):
    """The timestamp of the most recently modified file in the experiment.
    """
    files = [
        expt.path,
        expt.pipette_file,
        expt.nwb_file,
        expt.mosaic_file,
        expt.connections_file,
        os.path.join(expt.path, '.index'),
        os.path.join(os.path.split(expt.path)[0], '.index'), ## slice path
        os.path.join(os.path.split(os.path.split(expt.path)[0])[0], '.index'), ## expt_path
    ]
    mtime = 0
    for f in files:
        if f is None or not os.path.exists(f):
            continue
        mtime = max(mtime, os.stat(f).st_mtime)
    
    return timestamp_to_datetime(mtime)

def pipette_file(expt):
    """Return a pipettes.yml file for expt (or None)."""
    pf = os.path.join(expt.path, 'pipettes.yml')
    if not os.path.isfile(pf):
        return None
    return pf


def get_ephys_file(expt):
    """Return the ephys file for this experiment."""
    p = expt.path
    files = glob.glob(os.path.join(p, '*.nwb'))
    if len(files) == 0:
        files = glob.glob(os.path.join(p, '*.NWB'))
    if len(files) == 0:
        return None
    elif len(files) > 1:
        ephys_file = None
        # multiple NWB files here; try using the file manifest to resolve.
        manifest = os.path.join(p, 'file_manifest.yml')
        if os.path.isfile(manifest):
            manifest = yaml.load(open(manifest, 'rb'))
            for f in manifest:
                if f['category'] == 'MIES physiology':
                    ephys_file = os.path.join(os.path.dirname(self.path), f['path'])
                    break
        if ephys_file is None:
            raise Exception("Multiple NWB files found for %s" % expt)
    else:
        ephys_file = files[0]
    return ephys_file

def get_mosaic_file(expt):
    if not os.path.exists(expt.path):
        return None
    sitefile = os.path.join(expt.path, "site.mosaic")
    if not os.path.isfile(sitefile):
        sitefile = os.path.join(os.path.split(expt.path)[0], "site.mosaic")
    if not os.path.isfile(sitefile):
        mosaicfiles = [f for f in os.listdir(expt.path) if f.endswith('.mosaic')]
        if len(mosaicfiles) == 1:
            sitefile = os.path.join(expt.path, mosaicfiles[0])
    if not os.path.isfile(sitefile):
        # print(os.listdir(self.path))
        # print(os.listdir(os.path.split(self.path)[0]))
        return None
    return sitefile

def load_from_file(expt, file_path):
    """First function that is called during expt initialization. Load information from the file at file_path
    to populate fields in expt. Must populate: electrodes, _cells

    For opto, file_path leads to a connections.json file. A file from either acq4's mosaic editor or new_test_ui.py is accepted.
    """
    expt.electrodes = OrderedDict()
    expt._cells = OrderedDict()

    filename = os.path.basename(file_path)
    expt._uid=filename.split('_connections')[0]
    expt.source_id = (filename, None)
    expt._connections_file = file_path

    with open(file_path,'r') as f:
        exp_json=json.load(f)
    version = exp_json.get('version', None)

    if version is None:
        load_markPoints_connection_file(expt, exp_json)
    else:
        load_mosaiceditor_connection_file(expt, exp_json)

def load_from_site_path(expt, site_path):
    cnx_file = get_connections_file(expt, expt.path)
    #print("attempting to load ", cnx_file)
    load_from_file(expt, cnx_file)

    




def process_meta_info(expt, meta_info):
    """Process optional meta_info dict that is passed in at the initialization of expt.
    Called after load_from_file."""
    ## Need to load: presynapticCre, presynapticEffector, [class, reporter, layer for each headstage], internal

    preEffector = meta_info.get('presynapticEffector', '').lower()
    for e_id, elec in expt.electrodes.items():
        n = e_id[-1]
        cell = elec.cell
        cell._morphology['initial_call'] = meta_info.get('HS%s_class'%n)
        cell._target_layer = meta_info.get('HS%s_layer'%n)
        if meta_info.get('HS%s_reporter'%n, '').lower() == 'positive':
            cell._cre_type = meta_info.get('presynapticCre','').lower()
        label_cell(cell, preEffector, meta_info.get('HS%s_reporter'%n, '').lower() == 'positive')
        elec._internal_solution = meta_info.get('internal', '').lower()
        if len(meta_info.get('distances', [])) > 0: ### we may not have distance measurements for all cells
            dist = [e for e in meta_info.get('distances') if e.get('headstage')==n]
            if len(dist) > 1:
                raise Exception('Something is wrong.')
            cell._distance_to_pia = float(dist[0]['toPia'])*1e-6
            cell._distance_to_wm = float(dist[0]['toWM'])*1e-6


    for i, cell in expt.cells.items():
        if not cell.has_readout and cell.has_stimulation:
            cell._cre_type = meta_info.get('presynapticCre', '').lower()
            label_cell(cell, preEffector, positive=True) ## assume all non-patched stimulated cells are positive for preEffector

    expt.expt_info ## just have to touch it 
    expt.expt_info['internal_solution'] = meta_info.get('internal', '').lower() 



def get_uid(expt):
    return expt._uid

##### Private functions: ######

def label_cell(cell, preEffector, positive=True):
    """Populate appropriate labels for a cell positive for the preEffector."""

    ## Wanted to separate this function out so it is easier to change/amend the logic later

    ## if the cell is positive for the preEffector, populate genetic labels
    if positive:
        cell.labels[preEffector] = True
        if preEffector == 'ai167':
            cell.labels['tdTomato'] = True
        elif preEffector == 'ai136':
            cell.labels['EYFP'] = True

    ## if the cell is patched (has an electrode), populate dyes 
    ##    -- this assumes which dye is used for the whole experiment based on the color of the preEffector
    if cell.electrode is not None:
        if preEffector == 'ai167':
            cell.labels['AF488'] = True
        elif preEffector == 'ai136':
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
        #for p, conn in data['Connections'].items():
        #    if conn is None: ## skip this one, it doesn't have a match in points and is a duplicate
        #        continue
        #    points[p][headstage] = conn
        for p in points.keys():
            points[p][headstage] = data['Connections'][p]



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
    # for p in expt.pairs.values():
    #     if p.preCell.cell_id in points:
    #         if p.postCell.cell_id in HS_keys:
    #             p._connection_call = points[p.preCell.cell_id][p.postCell.cell_id]

    populate_connection_calls(expt, exp_json)


def load_mosaiceditor_connection_file(expt, exp_json):
    ## create Cells for stimulation points
    for name, data in exp_json['StimulationPoints'].items():
        if data['onCell']:
            cell = Cell(expt, name, None)
            cell.position = tuple(data['position'])
            cell.has_readout = False
            cell.has_stimulation = True
            cell._target_layer = data.get('target_layer')
            cell._percent_depth = data.get('percent_depth')
            expt._cells[cell.cell_id] = cell

    ## create Cells for recorded cells
    for name, data in exp_json['Headstages'].items():
        elec = Electrode(name, start_time=None, stop_time=None, device_id=name[-1])
        expt.electrodes[name] = elec
        cell = Cell(expt, name, elec)
        elec.cell = cell
        cell.position = (data['x_pos'], data['y_pos'], data['z_pos'])
        cell.angle = data['angle']
        cell.has_readout = True
        cell.has_stimulation = True
        cell._target_layer = data.get('target_layer')
        cell._percent_depth = data.get('percent_depth')
        expt._cells[cell.cell_id] = cell

    d = expt.cortical_site_info.get('pia_to_wm_distance')
    if d is not None:
        for cell in expt.cells.values():
            if cell.percent_depth is not None:
                cell._distance_to_pia = cell.percent_depth * d
                cell._distance_to_wm = (1-cell.percent_depth) * d

    # ## populate pair values
    # for p in expt.pairs.values():
    #     try:
    #         p.connection_call = exp_json['Headstages'][p.postCell.cell_id]['Connections'][p.preCell.cell_id]
    #     except KeyError:
    #         print("Could not find connection call for Pair %s -> %s in experiment %s" % (p.preCell.cell_id, p.postCell.cell_id, expt.uid))

    populate_connection_calls(expt, exp_json)


def populate_connection_calls(expt, exp_json):
    for p in expt.pairs.values():
        try:
            p._connection_call = exp_json['Headstages'][p.postCell.cell_id]['Connections'][p.preCell.cell_id]
            p._probed = True
        except KeyError:
            p._probed = False
            #print("Could not find connection call for Pair %s -> %s in experiment %s" % (p.preCell.cell_id, p.postCell.cell_id, expt.uid))


def get_connections_file(expt, site_path):
    cnx_files = sorted(glob.glob(os.path.join(site_path, '*connections*.json')))
    #print('cnx_files:', cnx_files, "path:", site_path)
    if len(cnx_files) == 1:
        return cnx_files[0]
    elif len(cnx_files) == 0:
        raise Exception("Could not find a connections file.")
    else:
        ### return the file with the highest version number. If there's still more than one return the file with the latest modification time
        max_version = 0
        cnx_file = [cnx_files[0]]
        for f in cnx_files:
            with open(f, 'r') as f2:
                exp_json = json.load(f2)
            version = exp_json.get('version', 0)
            if version > max_version:
                max_version = version
                cnx_file = [f]
            elif version == max_version:
                cnx_file.append(f)

        if len(cnx_file) == 1:
            return cnx_file[0]
        else:
            mts = map(os.path.getmtime, cnx_file)
            i = mts.index(max(mts))
            return cnx_file[i]

        #raise Exception("Need to implement choosing which file to load. Options are %s" %str(cnx_files))
            
    
def cortical_site_info(expt):
    with open(expt.connections_file,'r') as f:
        exp_json=json.load(f)
    version = exp_json.get('version', None)

    if version >= 3:
        return exp_json['CortexMarker']

    else:
        return {}








