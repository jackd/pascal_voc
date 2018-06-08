"""dids interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dids.core
from .core import get_voc_data


class PascalVocDataset(dids.core.UnwritableDataset):
    def __init__(self, key='base', mode='train', **kwargs):
        self._data = get_voc_data(key, mode, **kwargs)
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


merge_datasets = dids.core.PrioritizedDataset


# def get_combined_dataset(mode):
#     """
#     Get a combined dataset made up of relevant sub-datasets.
#
#     Args:
#         mode: one of 'train', 'val'. If 'train', combines base train and
#             augmented val. If 'val' uses non-overlapping elements of base val
#
#     Returns dataset based on zip file in ./_zipped. Creats the dataset first
#         if not already there. Highly redundant... but sometimes zip beat tar
#     """
#     path = os.path.join(
#         os.path.realpath(os.path.dirname(__file__)),
#         '_data', 'data.zip' % mode)
#     if not os.path.isfile(path):
#         _create_combined_data()
#     return _ZippedSegmentationDataset(path, mode)
