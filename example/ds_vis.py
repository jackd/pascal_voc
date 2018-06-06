#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from pascal_voc.dataset import PascalVocDataset
# from pascal_voc.seg import get_colors, image_to_seg

# colors = get_colors()

with PascalVocDataset(mode='train', year=2012) as ds:
    for key in ds:
        example = ds[key]
        seg = example.load_class_segmentation()
        im = example.load_image()
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(im)
        ax1.imshow(seg)
        ax2.imshow(np.array(seg))
        plt.show()
