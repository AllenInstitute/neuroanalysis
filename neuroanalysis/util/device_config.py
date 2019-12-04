from aisynphys import lims
import re


serial_number_to_rig = {
    4284: 'Wayne', # prairie
    831400: 'Wayne', # amplifier 1
    837041: 'Wayne', # amplifier 2
    4352: 'Garth', # prairie
    830774: 'Garth', # amplifier 1
    830775: 'Garth' # amplifier 2
}

device_mapping = {
    'Wayne': {
        'AD0': 'Electrode_0',
        'AD1': 'Electrode_1',
        'AD2': 'Electrode_2',
        'AD3': 'Electrode_3',
        'AD6': 'Fidelity',
        'TTL1_0': 'Prairie_Command',
        'TTL1_1': 'LED-470nm',
        'TTL1_2': 'LED-590nm'
    }
}


def find_lims_specimen_ids(slice_dh):
    """Return a list of lims specimen_ids that match the metainfo in the day and slice .index files.
    Search order:
        1) slice.specimen_ID
        2) day.animal_ID + slice.slice_number

    """

    info = slice_dh.info()
    parent_info = slice_dh.parent().info()

    sid = info.get('specimen_ID', '').strip()
    if sid == '':
        slice_id = info.get('slice_number', '').strip()
        if len(slice_id) > 2:
            sid = slice_id
        else:
            animal_id = parent_info.get('animal_id', '').strip()
            if len(animal_id) == 0:
                animal_id = parent_info.get('animal_ID', '').strip()
                if len(animal_id) == 0:
                    return []
            if len(slice_id) == 1:
                slice_id = '0'+slice_id
            sid = animal_id + '.' + slice_id

    #print('sid:', sid)
    ids = lims.find_specimen_ids_matching_name(sid)
    if len(ids) == 1:
        return ids

    possible_ids = []
    for n in ids:
        r = lims.query("select specimens.name as specimen_name from specimens where specimens.id=%d"%n)
        if len(r) != 1:
            raise Exception("LIMS lookup for specimen '%s' returned %d results (expected 1)" % (str(n), len(r)))
        rec = dict(r[0])
        m = re.match(r'(.*)(-(\d{6,7}))?(\.(\d{2}))(\.(\d{2}))$', rec['specimen_name'])
        if m is not None:
            possible_ids.append(n)

    return possible_ids


def get_rig_name_from_serial_number(sn):
    """Get a rig name from a given serial number. Serial number must be in the serial_number_to_rig dict."""

    global serial_number_to_rig
    rig = serial_number_to_rig.get(int(sn), None)

    if rig is None:
        raise Exception('No registry for serial number: %i' % int(sn))

    return rig

def get_rig_from_nwb(nwb=None, notebook=None):
    """Look up serial numbers in nwb to determine which rig this was recorded on. Uses serial_number_to_rig dict."""
    if notebook is not None:
        nb = notebook
    else:
        nb = nwb.notebook()
    ## serial number is recorded in many places, make sure they converge on one rig
    sns = []
    for sweeps in nb.values():
        for channel in sweeps:
            sn = channel.get('Serial Number', None)
            if sn is not None:
                sns.append(sn)
    unique_sns = list(set(sns))
    rigs = []
    for sn in unique_sns:
        rigs.append(get_rig_name_from_serial_number(sn))
    unique_rigs = list(set(rigs))
    if len(unique_rigs) != 1:
        raise Exception("Could not resolve rig for experiment %s. Found %s" %(expt.uid, unique_rigs))
    rig = unique_rigs[0]
    return rig



