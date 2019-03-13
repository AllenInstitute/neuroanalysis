

class Pair(object):

    """Represents a unidirectional possible connection between two cells.

    Any two cells will have 2 Pair objects:
        Pair1 = Pair(cellA, cellB)
        Pair2 = Pair(cellB, cellA)

    A pair may have been probed or not, have a connection or not."""

    def __init__(self, preCell, postCell):
        self.preCell = preCell
        self.postCell = postCell

    def wasProbed(self):
        """Return True if the connection from self.preCell -> self.postCell was 
        tested for connectivity"""

    def isConnected(self):
        """Return True if the two cells are connected by either a synapse or a gap junction."""
        return self.isSynapse() or self.isGapJunction()

    def isSynapse(self):
        """Return True if there is a synapse between preCell and postCell."""

    def isGapJunction(self):
        """Return True if there is a gap junction between preCell and postCell."""
