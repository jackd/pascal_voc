#!/usr/bin/python
from pascal_voc.read_tar import TarData

for mode in ('train', 'val', 'trainval'):
    with TarData(mode=mode) as data:
        print('%s: %d' % (mode, len(data.get_example_ids())))
