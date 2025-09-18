from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil
import cv2
from torchvision import transforms
from utils import readlines
#from kitti_utils import generate_depth_map


def export_gt_depths():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the SCARED data',
                        required=True)  # 添加required=True确保必须有此参数
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        choices=["endovis"],
                        default="endovis")
    opt = parser.parse_args()  # 移除args=[]以正确解析命令行参数

    if opt.data_path is None:
        raise ValueError("--data_path argument is required")
    
    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))
    print("Exporting ground truth depths for {}".format(opt.split))
    i=0
    gt_depths = []
    for line in lines:
        i = i+1
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)
        print(i)
        print(folder)

        if opt.split == "endovis":
            f_str = "scene_points{:06d}.tiff".format(frame_id - 1)
            gt_depth_path = os.path.join(
                opt.data_path,
                folder,
                "data",
                "scene_points",
                f_str)
            depth_gt = cv2.imread(gt_depth_path, 3)
            depth_gt = depth_gt[:, :, 0]
            depth_gt = depth_gt[0:1024, :]

        gt_depths.append(depth_gt.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths()
