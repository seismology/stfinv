#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GPLv3 License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "stfinv: Invert for seismic source time functions"
# Long description will go up on the pypi page
long_description = """

STFinv
========
STFinv is an ObsPy-based tool to invert for seismic source solutions including
earthquake depth, moment tensor and source time function using teleseismic body
waves, especially P and SH.

License
=======
``STFinv`` is licensed under the terms of the GPL version 3. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2016 - Simon Staehler (mail@simonstaehler.com)
"""

NAME = "stfinv"
MAINTAINER = "Simon Staehler"
MAINTAINER_EMAIL = "mail@simonstaehler.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/seismology/stfinv"
DOWNLOAD_URL = ""
LICENSE = "GPLv3"
AUTHOR = "Simon Staehler"
AUTHOR_EMAIL = "mail@simonstaehler.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['stfinv',
            'stfinv.tests',
            'stfinv.utils']
PACKAGE_DATA = {'stfinv': [pjoin('data', '*')]}
REQUIRES = ['numpy', 'obspy', 'instaseis', 'cartopy']
