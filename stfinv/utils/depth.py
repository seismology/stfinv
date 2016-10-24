#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  depth.py
#   Purpose:   Provide class for results of one depth in stfinv
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

from stfinv.utils.iteration import Iteration
import pickle


class Depth():
    def __init__(self, iterations=None):
        self.iterations = []
        if isinstance(iterations, Iteration):
            iterations = [iterations]
        if iterations:
            self.iterations.extend(iterations)

    def __iter__(self):
        return list(self.iterations).__iter__()

    def __setitem__(self, index, iteration):
        """
        __setitem__ method of stfinv.Results objects.
        """
        self.iterations.__setitem__(index, iteration)

    def __getitem__(self, index):
        """
        __getitem__ method of stfinv.Results objects.

        :return: iteration objects
        """
        if isinstance(index, slice):
            return self.__class__(
                iterations=self.iterations.__getitem__(index))
        else:
            return self.iterations.__getitem__(index)

    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of iterations.
        """
        return self.iterations.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        """
        __getslice__ method of obspy.Stream objects.

        :return: Results object
        """
        # see also https://docs.python.org/3/reference/datamodel.html
        return self.__class__(
            iterations=self.iterations[max(0, i):max(0, j):k])

    def append(self, iteration):
        """
        Append a single Trace object to the current Stream object.

        :param iteration: :class:`~stfinv.utils.Iteration` object.

        .. rubric:: Example

        """
        if isinstance(iteration, Iteration):
            self.iterations.append(iteration)
        else:
            msg = 'Append only supports a single Iteration object \
                   as an argument.'
            raise TypeError(msg)
        return self

    def get_best_solution(self):
        it_best = self[1]
        for it in self[1:]:
            if it.misfit < it_best.misfit:
                it_best = it

        return it_best

    def save(self, fnam):
        """
        Save results to pickle on disk

        Keywords:
        :type  fnam: string
        :param fnam: filename to save stack into
        """
        pickle.dump(self, open(fnam, 'wb'))
