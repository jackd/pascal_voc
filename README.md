Python loader functions for PASCAL VOC base and augmented datasets.

### Setup
* Download the [base dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and/or the [augmented dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). The rest of these instructions assume they're saved at `~/voc_data/`, i.e. you have `~/voc_data/VOCdevkit_18-May-2011.tar` and/or `~/voc_data/benchmark.tgz`. Note: you do not need to extract the contents.
* Specify where you have put these files on the environment variable `PASCAL_VOC_PATH`:
```
export PASCAL_VOC_PATH=~/voc_data
```
* Clone this repository and add the parent directory to your `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/pascal_voc.git
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```

### Acknowledgements
Please respect the work done by the dataset creators and cite their work.

Base dataset:
```
@misc{pascal-voc-2012,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}
```

Augmented dataset:
```
@InProceedings{BharathICCV2011,
  author       = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and Subhransu Maji and Jitendra Malik",
  title        = "Semantic Contours from Inverse Detectors",
  booktitle    = "International Conference on Computer Vision (ICCV)",
  year         = "2011",
}
```
