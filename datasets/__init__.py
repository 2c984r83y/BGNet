# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from .kitti_dataset import KITTIDataset
from .kitti12_dataset import KITTI_12_Dataset
from .dsec_png_dataset import DSEC_png_Dataset
from .dsec_png_batch_dataset import DSEC_png_batch_Dataset
from .dsec_pt_dataset import DSEC_pt_Dataset

__datasets__ = {
    "kitti": KITTIDataset,
    "kitti_12": KITTI_12_Dataset,
    "dsec_png": DSEC_png_Dataset,
    "dsec_png_batch": DSEC_png_batch_Dataset,
    "dsec_pt":DSEC_pt_Dataset
}
