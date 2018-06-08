#!/usr/bin/python

from pascal_voc.dataset import PascalVocDataset
import matplotlib.pyplot as plt
import numpy as np

with PascalVocDataset('combined', 'train') as ds:
    for key in ds:
        example = ds[key]
        seg = example.load_class_segmentation()
        im = example.load_image()
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(im)
        ax1.imshow(seg)
        seg = np.array(seg)
        seg[seg == 255] = 0
        ax2.imshow(np.array(seg))
        plt.show()
