from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


def _check_in(key, value, values):
    if value not in values:
        raise ValueError('%s must be in %s, got %s' % (key, values, value))


problems = frozenset(('Segmentation', 'Main', 'Layout', 'Action'))
keys = frozenset(('base', 'augmented', 'combined'))

classes = (
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)


def get_pascal_voc_dir():
    key = 'PASCAL_VOC_PATH'
    if key in os.environ:
        dataset_dir = os.environ[key]
        if not os.path.isdir(dataset_dir):
            raise Exception('%s directory does not exist' % key)
        return dataset_dir
    else:
        raise Exception('%s environment variable not set.' % key)


def get_tar_file(tar, subpath):
    return tar.extractfile(tar.getmember(subpath))


def load_tar_image(tar, subpath):
    from PIL import Image
    return Image.open(get_tar_file(tar, subpath))


def load_zip_image(zip, subpath):
    from PIL import Image
    with zip.open(subpath, 'r') as fp:
        return Image.open(fp)


class VocData(object):
    """Abstract base class for all PASCAL VOC data."""
    def __init__(self, mode):
        self._mode = mode
        self._data = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        if self.is_open:
            raise RuntimeError('TarData already open')
        self._data = self._load_data()
        return self

    def close(self):
        if self.is_closed:
            raise RuntimeError('TarData already closed')
        self._data.close()
        self._data = None

    @property
    def is_closed(self):
        return self._data is None

    @property
    def is_open(self):
        return not self.is_closed

    def _load_data(self):
        raise NotImplementedError('Abstract method')

    def get_example_ids(self, problem='Segmentation'):
        raise NotImplementedError('Abstract method')

    def get_example(self, example_id):
        raise NotImplementedError('Abstract method')


class VocExample(object):
    def __init__(self, voc_data, example_id):
        self._voc_data = voc_data
        self._example_id = example_id

    @property
    def example_id(self):
        return self._example_id

    def load_class_segmentation(self):
        raise NotImplementedError('Abstract method')

    def load_object_segmentation():
        raise NotImplementedError('Abstract method')

    def load_image(self):
        raise NotImplementedError('Abstract method')


def get_voc_data(key='base', mode='train', **kwargs):
    _check_in('key', key, keys)
    if key == 'base':
        from .base import BaseVocData
        return BaseVocData(mode=mode, **kwargs)
    elif key == 'augmented':
        from .augmented import AugmentedVocData
        return AugmentedVocData(mode=mode, **kwargs)
    elif key == 'combined':
        from .combined import CombinedVocData
        return CombinedVocData(mode=mode, **kwargs)
    else:
        raise ValueError('key "%s" not recognized' % key)
