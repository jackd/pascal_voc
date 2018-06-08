"""
Based on https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae

Python implementation of the color map function for the PASCAL VOC data set.
Official Matlab version can be found in the PASCAL VOC devkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
from .core import labels


def image_to_seg(image, colormap, dtype=np.int32):
    if image.shape[2:] != colormap.shape[1:]:
        raise ValueError(
            'Shape inconsistencies: %s, %s'
            % (str(image.shape), str(colormap.shape)))
    seg = np.zeros(shape=image.shape[:2], dtype=dtype)
    for i, c in enumerate(colormap):
        seg[image == c] = i+1
    return seg


def get_colors(N=None):
    if N is None:
        N = len(labels)

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    def get_color(c):
        r = g = b = 0
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        return r, g, b

    cmap = np.array(tuple(get_color(i) for i in range(N)), dtype=np.uint8)

    return cmap


def color_map_viz():
    nclasses = len(labels)
    row_size = 50
    col_size = 500
    cmap = get_colors(nclasses)
    array = np.empty(
        (row_size*(nclasses), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]

    imshow(array)
    plt.yticks([row_size*i+row_size//2 for i in range(nclasses)], labels)
    plt.xticks([])
    plt.show()
