# helpful tools for handling Qt signals

def disconnect(signal, slot):
    """Disconnect a Qt signal from a slot.

    This method augments Qt's Signal.disconnect():

    * Return bool indicating whether disconnection was successful, rather than
      raising an exception
    * Attempt to disconnect prior versions of the slot when using pg.reload    
    """
    try:
        signal.disconnect(slot)
        return True
    except (TypeError, RuntimeError):
        return False


class SignalBlock(object):
    """Class used to temporarily block a Qt signal connection::

        with SignalBlock(signal, slot):
            # do something that emits a signal; it will
            # not be delivered to slot
    """
    def __init__(self, signal, slot):
        self.signal = signal
        self.slot = slot

    def __enter__(self):
        self.reconnect = disconnect(self.signal, self.slot)
        return self

    def __exit__(self, *args):
        if self.reconnect:
            self.signal.connect(self.slot)

