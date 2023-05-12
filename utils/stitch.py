# -*- coding: utf-8 -*-

"""Stitching the patches.

This script stitches the patches to the original size of the image.

Example:
    $ python stitch.py --patch_path ./patches --save_path ./stitched
"""

__author__ = "Mir Sazzad Hossain"


import argparse
import os

from PIL import Image


def stitch_patches(patch_path: str, save_path: str) -> None:
    """
    Stitch the patches.

    :param patch_path: str, patch path.
    :type patch_path: str
    :param save_path: str, save path.
    :type save_path: str
    """
    # Get the list of images as subdirectories.
    image_list = os.listdir(patch_path)

    # Each subdirectory contains the patches of one image.
    for image in image_list:
        patches = os.listdir(os.path.join(patch_path, image))
        bbox = []
        patch_image = []
        for patch in patches:
            bbox.append(
                [
                    int(patch.split(".")[0].split("_")[i]) for i in range(4)
                ]
            )
            patch_image.append(
                Image.open(os.path.join(patch_path, image, patch))
            )

        # Sort the patches according to the bbox.
        bbox, patch_image = zip(*sorted(zip(bbox, patch_image)))

        # Stitch the patches.
        out_image = Image.new("RGB", (bbox[-1][2], bbox[-1][3]))
        for i in range(len(patches)):
            out_image.paste(patch_image[i], bbox[i])

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the stitched image.
        out_image.save(os.path.join(save_path, image + ".png"))


if __name__ == "__main__":
    # Get the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    # Stitch the patches.
    stitch_patches(args.patch_path, args.save_path)
