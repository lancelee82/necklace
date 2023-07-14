""""""
import time


class TimerItLogger(object):
    def __init__(self, cfg={}, *args, **kwargs):
        self.cfg = cfg

        self.do_print = self.cfg.get('do_print', False) or kwargs.get('do_print', False)

        self.tms_d = {}

    def start(self, key):
        if not self.do_print:
            return

        tm = time.time()
        self.tms_d[key] = [tm, tm]

    def end(self, key):
        if not self.do_print:
            return

        tm = time.time()
        self.tms_d[key][1] = tm

        print('tm::', key, ':', self.tms_d[key][1] - self.tms_d[key][0])
