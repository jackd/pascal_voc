from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from .core import VocData, VocExample, get_pascal_voc_dir
from .core import _check_in, load_tar_image, get_tar_file
from scipy.io import loadmat


class AugmentedVocData(VocData):
    def __init__(self, mode='train'):
        _check_in('mode', mode, AugmentedVocData.modes)
        super(AugmentedVocData, self).__init__(mode)

    modes = frozenset(('train', 'val'))
    base_subpath = 'benchmark_RELEASE/dataset'

    def get_example_ids(self, problem='Segmentation'):
        subpath = os.path.join(
            AugmentedVocData.base_subpath, '%s.txt' % self._mode)
        member = self._data.getmember(subpath)
        fp = self._data.extractfile(member)
        ids = tuple(l.rstrip() for l in fp.readlines())[:-1]
        return ids

    def get_example(self, example_id):
        return AugmentedVocExample(self, example_id)

    def _load_data(self):
        path = os.path.join(get_pascal_voc_dir(), 'benchmark.tgz')
        if not os.path.isfile(path):
            raise ValueError('No augmented data found at %s' % path)
        return tarfile.TarFile(path, mode='r')


class AugmentedVocExample(VocExample):

    def load_class_segmentation(self):
        path = os.path.join(
            AugmentedVocData.base_subpath, 'cls', '%s.mat' % self._example_id)
        fp = get_tar_file(self._voc_data._data, path)
        return loadmat(fp)['GTcls']['Segmentation'][0][0]

    def load_object_segmentation(self):
        path = os.path.join(
            AugmentedVocData.base_subpath, 'inst', '%s.mat' % self._example_id)
        fp = get_tar_file(self._voc_data._data, path)
        insts = loadmat(fp)['GTcls'][0][0]
        return insts['Segmentation'], insts['Categories']

    def load_image(self):
        path = os.path.join(
            AugmentedVocData.base_subpath, 'img', '%s.jpg' % self._example_id)
        return load_tar_image(self._voc_data._data, path)
