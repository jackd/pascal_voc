from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import os
from .core import VocData, VocExample, get_pascal_voc_dir, load_tar_image
from .core import _check_in, problems


class BaseVocData(VocData):
    """Higher level class for data loading."""
    def __init__(self, year=2012, mode='train'):
        _check_in('year', year, BaseVocData.years)
        _check_in('mode', mode, BaseVocData.modes)
        self._year = year
        self._init_subpath = os.path.join('VOCdevkit', 'VOC%d' % year)
        super(BaseVocData, self).__init__(mode)

    modes = frozenset(('train', 'val', 'trainval'))
    years = (2012,)

    def _load_data(self):
        path = os.path.join(
            get_pascal_voc_dir(), 'VOCtrainval_11-May-2012.tar')
        if not os.path.isfile(path):
            raise ValueError('No base data found at %s' % path)
        return tarfile.TarFile(path, mode='r')

    def get_example_ids(self, problem='Segmentation'):
        _check_in('problem', problem, problems)
        subpath = os.path.join(
            self._init_subpath, 'ImageSets', problem, '%s.txt' % self._mode)
        member = self._data.getmember(subpath)
        fp = self._data.extractfile(member)
        ids = tuple(l.rstrip() for l in fp.readlines())[:-1]
        return ids

    def _get_segmentation_subpath(self, subdir, example_id):
        return '%s/%s/%s.png' % (self._init_subpath, subdir, example_id)

    def get_class_segmentation_subpath(self, example_id):
        return self._get_segmentation_subpath('SegmentationClass', example_id)

    def get_object_segmentation_subpath(self, example_id):
        return self._get_segmentation_subpath('SegmentationObject', example_id)

    def get_image_subpath(self, example_id):
        return '%s/JPEGImages/%s.jpg' % (self._init_subpath, example_id)

    def load_tar_image(self, subpath):
        return load_tar_image(self._data, subpath)

    def load_class_segmentation(self, example_id):
        return self.load_tar_image(
            self.get_class_segmentation_subpath(example_id))

    def load_object_segmentation(self, example_id):
        return self.load_tar_image(
            self.get_object_segmentation_subpath(example_id))

    def load_image(self, example_id):
        return self.load_tar_image(self.get_image_subpath(example_id))

    def get_example(self, example_id):
        return BaseVocExample(self, example_id)


class BaseVocExample(VocExample):
    def load_class_segmentation(self):
        return self._voc_data.load_class_segmentation(self._example_id)

    def load_object_segmentation(self):
        return self._voc_data.load_object_segmentation(self._example_id)

    def load_image(self):
        return self._voc_data.load_image(self._example_id)
