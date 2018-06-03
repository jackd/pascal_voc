from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def get_pascal_voc_dir():
    key = 'PASCAL_VOC_PATH'
    if key in os.environ:
        dataset_dir = os.environ[key]
        if not os.path.isdir(dataset_dir):
            raise Exception('%s directory does not exist' % key)
        return dataset_dir
    else:
        raise Exception('%s environment variable not set.' % key)


def get_data_path():
    return os.path.join(get_pascal_voc_dir(), 'VOCtrainval_11-May-2012.tar')
