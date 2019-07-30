from __future__ import print_function, division
import numpy as np

from multipatch_analysis.constants import ALL_CRE_TYPES, ALL_LABELS, EXCITATORY_CRE_TYPES, INHIBITORY_CRE_TYPES


class Cell(object):
    """Represents a single cell recorded during an experiment.

    Parameters
    ----------
    expt : Experiment
        The experiment that this cell is included in
    cell_id : int
        ID that identifies this cell uniquely amongst all other cells in the same experiment.

    """
    def __init__(self, expt, cell_id, electrode):
        self.expt = expt
        self.cell_id = cell_id
        self.electrode = electrode
        self.access_qc = None
        self.holding_qc = None
        self.spiking_qc = None
        self._morphology = {}
        self.labels = {}
        self._raw_labels = {}
        self.position = None
        self._target_layer = None
        self._percent_depth = None
        self._is_excitatory = None
        self._cre_type = None
        self.has_readout = None ## should be set to True if we can have info about this cells postsynaptic activity, False if we don't
        self.has_stimulation = None ## should be set to True if we can have info about this cells presynaptic activity, False if we don't
                                    ## examples:
                                    ##      patched cell:           has_readout = True, has_stimulation = True
                                    ##      photostimulated cell:   has_readout = False, has_stimulation = True
                                    ##      VSD imaged cell:        has_readout = True, has_stimulation = True ## even though we may not be stimulating, we have informationa about the presynaptic activity

    @property
    def pass_qc(self):
        """True if cell passes QC.
        """
        if self.access_qc is True and self.holding_qc is True:
            return True
        elif self.access_qc is False or self.holding_qc is False:
            return False

        # None means cell is not present in ephys data
        return None

    @property
    def cell_type(self):
        """Dict describing the type of this cell like::

            {
                'layer': '5a',
                'driver': 'tlx3',
                'morphology': 'pyramidal',
                'excitatory': True,
            }

        All keys are optional.
        """
        typ = {}
        cre = self.cre_type
        if cre is not None:
            typ['driver'] = cre
        layer = self.target_layer
        if layer is not None:
            typ['layer'] = layer
        return typ

    @property
    def morphology(self):
        """Dictionary of morphology metrics for this cell.

            {'pyramidal': True, 'spiny': True}
        """
        return self._morphology.copy()

    @property
    def cre_type(self):
        if self._cre_type is None:
            return self.deduce_cre_type()
        else:
            return self._cre_type

    def deduce_cre_type(self):
        """Cre type string for this cell.
        
        If the cell is positive for multiple cre types, then they will be returned
        as a comma-separated string.
        If the cell is reporter-negative then cre_type is 'unknown'.
        If the cell has ambiguous or missing data then cre_type is None.
        """
        default = 'unknown'
        ct = []
        for label,pos in self.labels.items():
            if label not in ALL_CRE_TYPES:
                continue
            if pos in ('+', True):
                ct.append(label)
        if len(ct) == 0:
            return default
        return ','.join(ct)

    @property
    def label_type(self):
        """fluorescent type string for this cell.
        
        If the cell is reporter-negative then cre_type is 'unk'.
        If the cell has ambiguous or missing data then cre_type is None.
        """
        default = 'unknown'
        ct = None
        for label,pos in self.labels.items():
            if label in ALL_CRE_TYPES or label == 'biocytin':
                continue
            if pos == '+':
                if ct not in (None, default):
                    raise Exception("%s has multiple labels!" % self)
                ct = label
            elif pos == '-':
                if ct is not None:
                    continue
                ct = default
        return ct

    @property
    def target_layer(self):
        """Intended cortical layer for this cell; actual layer call may be
        different.
        """
        return self._target_layer

    @property
    def percent_depth(self):
        """Columnar depth where 0 is the top of the pia, and 1 is the bottom of L6."""
        return self._percent_depth

    @property
    def is_excitatory(self):
        if self._is_excitatory is None:
            # try to infer excitatory from cre line
            ct = self.cre_type
            if ct in EXCITATORY_CRE_TYPES:
                self._is_excitatory =  True
            elif ct in INHIBITORY_CRE_TYPES:
                self._is_excitatory = False
        return self._is_excitatory

    @property
    def depth(self):
        """Depth of cell from the cut surface of the slice.
        """
        sd = self.expt.surface_depth
        p = self.position
        if None in (sd, p):
            return None
        return sd - p[2]

    def distance(self, cell):
        """Return distance between cells, or nan if positions are not defined.
        """
        p1 = self.position
        p2 = cell.position
        if p1 is None or p2 is None:
            return np.nan
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

    def __repr__(self):
        return "<Cell %s:%s>" % (self.expt.source_id, str(self.cell_id))