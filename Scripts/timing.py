# -*- coding: utf-8 -*-

import time
from collections import defaultdict

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        Timing.times[f.__name__] += (time2-time1)
        return ret
    return wrap

class Timing(object):
    times = defaultdict(float)

    @classmethod
    def print_times(cls):
        for function, time in cls.times.items():
            print('%25s \t %0.3fs' % (function, time))
