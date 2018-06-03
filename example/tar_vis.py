from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from pascal_voc.read_tar import TarData
# from pascal_voc.seg import get_colors, image_to_seg

# colors = get_colors()

with TarData(mode='train') as data:
    example_ids = data.get_example_ids()
    for example_id in example_ids:
        example = data.get_example(example_id)
        seg = example.load_class_segmentation()
        im = example.load_image()
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(im)
        ax1.imshow(seg)
        ax2.imshow(np.array(seg))
        plt.show()
