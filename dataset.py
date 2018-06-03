"""dids interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dids.core
from .read_tar import TarData


class PascalVocDataset(dids.core.UnwritableDataset):
    def __init__(self, mode='train', year=2012):
        self._data = TarData(mode=mode, year=year)
        self._keys = None

    @property
    def is_open(self):
        return self._data.is_open

    def keys(self):
        self._assert_open('Cannot get keys from closed dataset')
        return self._keys

    def __getitem__(self, key):
        return self._data.get_example(key)

    def _open_resource(self):
        self._data.open()
        self._keys = frozenset(self._data.get_example_ids())

    def _close_resource(self):
        self._data.close()
        self._keys = None

    def __contains__(self, key):
        self._assert_open('Cannot check membership of closed dataset')
        return key in self._keys
