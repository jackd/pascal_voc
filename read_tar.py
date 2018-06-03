from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import os
from PIL import Image
from .path import get_data_path
from . import years, problems, modes


def get_data_tar():
    return tarfile.TarFile(get_data_path(), mode='r')


def _check_in(key, value, values):
    if value not in values:
        raise ValueError('%s must be in %s, got %s' % (key, values, value))


def get_example_ids(tar, year, problem, mode):
    _check_in('year', year, years)
    _check_in('problem', problem, problems)
    _check_in('mode', mode, modes)
    subpath = os.path.join(
        'VOCdevkit', 'VOC%d', 'ImageSets', problem, '%s.txt' % mode)
    member = tar.getmember(subpath)
    fp = tar.extractfile(member)
    ids = tuple(l.rstrip() for l in fp.readlines())[:-1]
    return ids


def get_segmentation_subpaths(year, class_or_object, example_ids):
    class_or_object = class_or_object.lower()
    _check_in('year', years)
    _check_in('class_or_object', class_or_object, ('class', 'object'))
    subdir = 'SegmentationClass' if class_or_object == 'class' \
        else 'SegmentationObject'
    return tuple(os.path.join(
        'VOCdevkit', 'VOC%d' % year, subdir, '%s.png' % i)
        for i in example_ids)


def get_image_subpaths(year, example_ids):
    _check_in('year', years)
    return tuple(os.path.join(
        'VOCdevkit', 'VOC%d' % year, 'JPEGImages', '%s.jpg' % i)
        for i in example_ids)


def load_tar_image(tar, subpath):
    fp = tar.extractfile(tar.getmember(subpath))
    image = Image.open(fp)
    return image


class TarData(object):
    """Higher level class for data loading."""
    def __init__(self, year=2012, mode='train'):
        _check_in('year', year, years)
        _check_in('mode', mode, modes)
        self._year = year
        self._mode = mode
        self._init_subpath = os.path.join('VOCdevkit', 'VOC%d' % year)
        self._tar = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        if self.is_open:
            raise RuntimeError('TarData already open')
        self._tar = get_data_tar()
        return self

    def close(self):
        if self.is_closed:
            raise RuntimeError('TarData already closed')
        self._tar.close()
        self._tar = None

    @property
    def is_closed(self):
        return self._tar is None

    @property
    def is_open(self):
        return not self.is_closed

    def get_example_ids(self, problem='Segmentation'):
        _check_in('problem', problem, problems)
        subpath = os.path.join(
            self._init_subpath, 'ImageSets', problem, '%s.txt' % self._mode)
        member = self._tar.getmember(subpath)
        fp = self._tar.extractfile(member)
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
        return load_tar_image(self._tar, subpath)

    def load_class_segmentation(self, example_id):
        return self.load_tar_image(
            self.get_class_segmentation_subpath(example_id))

    def load_object_segmentation(self, example_id):
        return self.load_tar_image(
            self.get_object_segmentation_subpath(example_id))

    def load_image(self, example_id):
        return self.load_tar_image(self.get_image_subpath(example_id))

    def get_example(self, example_id):
        return Example(self, example_id)


class Example(object):
    def __init__(self, data, example_id):
        self._data = data
        self._example_id = example_id

    @property
    def example_id(self):
        return self._example_id

    def load_class_segmentation(self):
        return self._data.load_class_segmentation(self._example_id)

    def load_object_segmentation(self):
        return self._data.load_object_segmentation(self._example_id)

    def load_image(self):
        return self._data.load_image(self._example_id)
