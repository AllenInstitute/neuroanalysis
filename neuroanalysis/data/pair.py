

class Pair(object):

    """Represents a unidirectional possible connection between two cells.

    Any two cells will have 2 Pair objects:
        Pair1 = Pair(cellA, cellB)
        Pair2 = Pair(cellB, cellA)

    A pair may have been probed or not, have a connection or not."""

    def __init__(self, preCell, postCell):
        self.preCell = preCell
        self.postCell = postCell

        self._probed = None
        self._connection_call = None 


    def was_probed(self):
        """Return True if the connection from self.preCell -> self.postCell was 
        tested for connectivity"""
        if self._probed is None:
            return False
        else:
            return self._probed

    # def isConnected(self):
    #     """Return True if the two cells are connected by either a synapse or a gap junction."""
    #     return self.isSynapse() or self.isGapJunction()

    def isSynapse(self):
        """Return True if there is a synapse between preCell and postCell."""
        if self._connection_call == 'excitatory' or self._connection_call == 'inhibitory':
            return True
        else:
            return False

    # def isGapJunction(self):
    #     """Return True if there is a gap junction between preCell and postCell."""
    @property
    def distance(self):
        return self.preCell.distance(self.postCell)