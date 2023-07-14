"""Data Exchanger with MP"""

class DataExchanger(object):
    def __init__(self, cfg={}, exchange_fn=None):
        self.cfg = cfg
        self.exchange_fn = exchange_fn

    def exchange(self, *args, **kwargs):
        pass  # TODO:
