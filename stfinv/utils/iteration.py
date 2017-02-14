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
    def __init__(self, tensor, origin, stf, CC, dA, dT, arr_times, CClim, it,
                 depth, misfit, st_data, st_synth):

        self.tensor = tensor
        self.origin = origin
        self.stf = stf
        self.CC = CC
        self.dA = dA
        self.dT = dT
        self.CClim = CClim
        self.misfit = misfit
        self.it = it
        self.depth = depth
        self.st_data = st_data
        self.st_synth = st_synth

        # For some reason, arr_times becomes a tuple otherwise
        self.arr_times = dict()
        for key, value in arr_times.items():
            self.arr_times[key] = value

    def __str__(self):
        return 'Depth: %d m, iteration: %d, misfit: %5.3f' % \
            (self.depth, self.it, self.misfit)

    def plot(self, outdir='waveforms_best'):
        plot_waveforms(st_data=self.st_data,
                       st_synth=self.st_synth,
                       arr_times=self.arr_times,
                       CC=self.CC,
                       CClim=self.CClim,
                       dA=self.dA,
                       dT=self.dT,
                       outdir=outdir,
                       misfit=self.misfit,
                       iteration=self.it,
                       stf=self.stf,
                       origin=self.origin,
                       depth=self.depth,
                       tensor=self.tensor)
