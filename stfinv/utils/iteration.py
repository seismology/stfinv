#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  iteration.py
#   Purpose:   Provide class for result of one iteration in stfinv
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------
from .plotting import plot_waveforms


class Iteration():
    def __init__(self, tensor, stf, CC, dA, dT, it, depth, misfit,
                 st_data, st_synth):

        self.tensor = tensor
        self.stf = stf
        self.CC = CC
        self.dA = dA
        self.dT = dT
        self.misfit = misfit
        self.it = it
        self.depth = depth
        self.st_data = st_data
        self.st_synth = st_synth

    def __str__(self):
        return 'Depth: %d m, iteration: %d, misfit: %5.3f' % \
            (self.depth, self.it, self.misfit)

    def plot(self, outdir='waveforms_best'):
        plot_waveforms(self.st_data,
                       self.st_synth,
                       self.arr_times,
                       self.CC,
                       self.CClim,
                       self.dA,
                       self.dT,
                       outdir=outdir,
                       misfit=self.misfit,
                       iteration=self.it,
                       stf=self.stf,
                       tensor=self.tensor)
