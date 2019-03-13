

class Experiment(object):
    """Base class defining the structure of an experiment.

        has:
            cells
            data_sets
            pairs
            electrodes
            stimulation_sites

    Should be subclassed to define how these fields are populated."""

