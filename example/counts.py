#!/usr/bin/python
from pascal_voc.core import get_voc_data

for key, modes in (
        ('base', ('train', 'val', 'trainval')),
        ('augmented', ('train', 'val')),
        ('combined', ('train', 'val')),
        ):
    for mode in modes:
        with get_voc_data(key, mode=mode) as data:
            print('%s - %s: %d' % (key, mode, len(data.get_example_ids())))
