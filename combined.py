from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .core import VocData, VocExample, load_zip_image
from .dataset import PascalVocDataset, merge_datasets
import zipfile


# def load_image_from_file(f):
#     # from StringIO import StringIO
#     # return Image.open(StringIO(f.read()))
#     return Image.open(f)


_data_dir = os.path.join(os.path.dirname(__file__), '_data')
_zip_path = os.path.join(_data_dir, 'data.zip')


class CombinedVocData(VocData):
    def _load_data(self):
        return zipfile.ZipFile(_zip_path)

    def get_example_ids(self, problem='Segmentation'):
        if problem != 'Segmentation':
            raise ValueError('Only "Segmentation" valid, got %s' % problem)
        with self._data.open('%s.txt' % self._mode, 'r') as fp:
            ids = tuple(l.rstrip() for l in fp.readlines())[:-1]
        return ids

    def get_example(self, example_id):
        return CombinedVocExample(self, example_id)


class CombinedVocExample(VocExample):

        def load_class_segmentation(self):
            return load_zip_image(
                self._voc_data._data, 'cls/%s.png' % self._example_id)

        def load_object_segmentation():
            raise NotImplementedError(
                'Not saving object segmentations for combined.')

        def load_image(self):
            return load_zip_image(
                self._voc_data._data, 'img/%s.jpg' % self._example_id)


def create_combined_data():
    import numpy as np
    from PIL import Image
    from progress.bar import IncrementalBar
    base_train = PascalVocDataset('base', 'train')
    aug_train = PascalVocDataset('augmented', 'train')
    aug_val = PascalVocDataset('augmented', 'val')
    train_ds = merge_datasets(base_train, aug_train, aug_val)
    eval_ds = PascalVocDataset('base', 'val')
    data_dir = os.path.join(_data_dir, 'data')
    img_dir = os.path.join(data_dir, 'img')
    seg_dir = os.path.join(data_dir, 'cls')
    for d in (data_dir, img_dir, seg_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    def save_example(example):
        example_id = example.example_id
        img = example.load_image()
        seg = example.load_class_segmentation()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(seg, np.ndarray):
            seg = Image.fromarray(seg)
        img.save(os.path.join(img_dir, '%s.jpg' % example_id))
        seg.save(os.path.join(seg_dir, '%s.png' % example_id))

    with train_ds:
        train_path = os.path.join(data_dir, 'train.txt')
        keys = list(train_ds.keys())
        keys.sort()
        with open(train_path, 'w') as fp:
            fp.writelines(''.join('%s\n' % k for k in keys))

        print('Saving train data...')
        bar = IncrementalBar(max=len(keys))
        for example in train_ds.values():
            save_example(example)
            bar.next()
        bar.finish()

        train_keys = frozenset(keys)

    with eval_ds:
        eval_path = os.path.join(data_dir, 'val.txt')
        keys = [k for k in eval_ds.keys() if k not in train_keys]
        keys.sort()
        with open(eval_path, 'w') as fp:
            fp.writelines(''.join('%s\n' % k for k in keys))

        print('Saving eval data...')
        bar = IncrementalBar(max=len(keys))
        for example_id in keys:
            example = eval_ds[example_id]
            save_example(example)
            bar.next()
        bar.finish()


def convert_to_archive():
    import shutil
    data_dir = os.path.join(_data_dir, 'data')
    print('Creating archive...')
    shutil.make_archive(_zip_path[:-4], 'zip', data_dir)
    print('Removing source files...')
    shutil.rmtree(data_dir)


def create_combined_archive():
    create_combined_data()
    convert_to_archive()
