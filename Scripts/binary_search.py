# -*- coding: utf-8 -*-

import numpy as np
import math

# source: http://pythonfiddle.com/unimodal-maximum-problem/

def binary_search(low, high, f):
    values = np.repeat(-1.0, high - low + 1)
    num_eval = 0

    def get_value(idx):
        idx = int(idx)
        if values[idx] != -1:
            return values[idx]

        nonlocal num_eval
        num_eval += 1
        x = int(low + idx)
        value = f(x)
        values[idx] = value
        return value

    n = np.size(values)
    depth = 2
    mid = n/2
    left = mid - 1
    right = mid + 1
    found = False

    while not found:
        found = True

        if get_value(left) < get_value(mid):
            if n > depth:
                depth *= 2
                mid -= math.floor(n/depth)
            else:
                mid -= 1

            left = mid - 1
            right = mid + 1
            found = False

        if get_value(right) < get_value(mid):
            if n > depth:
                depth *= 2
                mid += math.floor(n/depth)
            else:
                mid += 1

            left = mid - 1
            right = mid + 1
            found = False

    return int(low + mid), num_eval
