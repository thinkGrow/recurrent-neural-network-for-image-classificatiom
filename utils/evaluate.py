# -*- coding: utf-8 -*-

"""Evaluation matrices.

This script calculates the evaluation matrices.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from psnr import calculate_psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stitched_path",
        type=str,
        required=True,
        help="Stitched image path."
    )
    parser.add_argument(
        "--hr_path",
        type=str,
        required=True,
        help="High resolution image path."
    )
    args = parser.parse_args()

    # Get the list of images.
    stitched_list = os.listdir(args.stitched_path)
    stitched_list.sort()
    hr_list = os.listdir(args.hr_path)
    hr_list.sort()
    hr_list = hr_list[800:]

    # Calculate psnr.
    tot_psnr = 0
    for i in range(len(stitched_list)):
        stitched_image = Image.open(
            os.path.join(args.stitched_path, stitched_list[i])
        )
        hr_image = Image.open(os.path.join(args.hr_path, hr_list[i]))
        psnr = calculate_psnr(
            np.array(stitched_image),
            np.array(hr_image),
            rgb_range=255
        )
        print(f"PSNR for {stitched_list[i]}: {psnr}")
        tot_psnr += psnr
    print("Avg. PSNR: ", tot_psnr / len(hr_list))
