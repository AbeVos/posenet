#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:42:57 2017

@author: abe
"""

import numpy as np
import numpy.linalg as linalg

def lerp(A, B, t):
    return (1.0 - t) * A + t * B

def distance(A, B):
    return np.abs(linalg.norm(A - B))

def normalize(A):
    norm = linalg.norm(A)
    if norm > 0:
        return A / norm
    else:
        return np.zeros((2))