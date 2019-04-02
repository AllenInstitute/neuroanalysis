
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
    to populate fields in expt. Must populate: 
    """

def process_meta_info(expt, meta_info):
    """Process optional meta_info dict that is passed in at the initialization of expt.
    Called after load_from_file. Can have any format."""

def get_uid(expt):
    """Optional - Return a unique id for the experiment. 

    Default is to return None, in which case expt will return the site timestamp formatted to 2 decimal places, which is
    very likely to be unique for any site."""
    return None

