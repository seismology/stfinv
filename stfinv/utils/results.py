#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  results.py
#   Purpose:   Provide class for results in stfinv
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

from stfinv.utils.depth import Depth
import pickle


class Results():
    def __init__(self, depths=None):
        self.depths = []
        if isinstance(depths, Depth):
            depths = [depths]
        if depths:
            self.depths.extend(depths)

    def __iter__(self):
        return list(self.depths).__iter__()

    def __setitem__(self, index, depth):
        """
        __setitem__ method of stfinv.Results objects.
        """
        self.depths.__setitem__(index, depth)

    def __getitem__(self, index):
        """
        __getitem__ method of stfinv.Results objects.

        :return: depth objects
        """
        if isinstance(index, slice):
            return self.__class__(depths=self.depths.__getitem__(index))
        else:
            return self.depths.__getitem__(index)

    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of depths.
        """
        return self.depths.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        """
        __getslice__ method of obspy.Stream objects.

        :return: Results object
        """
        # see also https://docs.python.org/3/reference/datamodel.html
        return self.__class__(depths=self.depths[max(0, i):max(0, j):k])

    def append(self, depth):
        """
        Append a single Trace object to the current Stream object.

        :param depth: :class:`~stfinv.utils.Depth` object.

        .. rubric:: Example

        """
        if isinstance(depth, Depth):
            self.depths.append(depth)
        else:
            msg = 'Append only supports a single Depth object \
                   as an argument.'
            raise TypeError(msg)
        return self

    def get_best_depth(self):
        depth_best = self[0]
        for depth in self:
            if depth.get_best_solution().misfit < \
                    depth_best.get_best_solution().misfit:
                depth_best = depth

        return depth_best

    def save(self, fnam):
        """
        Save results to pickle on disk

        Keywords:
        :type  fnam: string
        :param fnam: filename to save stack into
        """
        pickle.dump(self, open(fnam, 'wb'))


def load_results(fnam):
    """
    Load Results object from pickle on disk

    Keywords:
    :type  fnam: string
    :param fnam: filename from which to read stack
    """
    return pickle.load(open(fnam, 'rb'))
