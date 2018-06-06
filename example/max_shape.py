#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pascal_voc.dataset import PascalVocDataset

with PascalVocDataset(mode='val', year=2012) as ds:
    max_h, max_w = 0, 0
    for key in ds:
        example = ds[key]
        seg = example.load_class_segmentation()
        h, w = seg.size
        max_h = max(h, max_h)
        max_w = max(w, max_w)

print(max_h, max_w)
