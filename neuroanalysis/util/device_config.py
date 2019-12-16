import neuroanalysis.util.mies_nwb_parsing as parser
import datetime

#### have these dicts here for now, can move out into yaml file later
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
        '2019-12-15':{
            #'AD0': 'Electrode_0',
            #'AD1': 'Electrode_1',
            #'AD2': 'Electrode_2',
            #'AD3': 'Electrode_3',
            'AD6': 'Fidelity',
            'TTL1_0': 'Prairie_Command',
            'TTL1_1': 'LED-470nm',
            'TTL1_2': 'LED-590nm'
            }
        }
    }


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

def get_device_config(notebook):
    rig = get_rig_from_nwb(notebook=notebook)
    date = parser.igorpro_date(notebook[0][0]['TimeStamp']).date()

    date = find_most_recent_date(date, rig, post_hoc=True)

    return device_mapping[rig][date]

def find_most_recent_date(date, rig, post_hoc=False):

    cal_dates = list(device_mapping[rig].keys())
    #cal_dates = [datetime.strptime(d, 'yyyy-mm-dd') for d in cal_dates]
    cal_dates.sort()
    cal_dates.reverse()

    for d in cal_dates:
        if date >= datetime.datetime(int(d[:4]), int(d[5:7]), int(d[-2:])).date():
            return d

    ## no calibrations earlier than date were found - post_hod==True allows later calibrations to be used
    if not post_hoc:
        raise Exception('No calibration found for %s. Options are %s' %(date.strftime('%Y-%m-%d'), str([d.strftime('%Y-%m-%d') for d in cal_dates])))

    else: 
        return cal_dates[0]





