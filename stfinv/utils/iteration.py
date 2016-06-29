#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  iteration.py
#   Purpose:   Provide class for result of one iteration in stfinv
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------


class Iteration():
    def __init__(self, tensor, stf, CC, dA, dT, it, depth, misfit):
        self.tensor = tensor
        self.stf = stf
        self.CC = CC
        self.dA = dA
        self.dT = dT
        self.misfit = misfit
        self.it = it
        self.depth = depth

    def __str__(self):
        return 'Depth: %d m, iteration: %d, misfit: %5.3f' % \
            (self.depth, self.it, self.misfit)
