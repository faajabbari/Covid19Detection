import sys

import argparse
import pydicom
import cv2
import numpy as np


def read_dicom(source_folder):

    ds = pydicom.dcmread(source_folder)
    image_2d = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    # Convert to uint8
    image_2d_scaled = np.uint8(image_2d_scaled)
    print(np.unique(image_2d_scaled))
    return image_2d_scaled

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dicom", default='')
    parser.add_argument("--mask_dicom", default='')
    args = parser.parse_args()
                            
    a = read_dicom(args.original_dicom)
    b = read_dicom(args.mask_dicom)
    #b = np.where(b!=0, 1, b)
    cv2.imwrite('a.png', a)
    cv2.imwrite('b.png', b)
    c = a * b
    cv2.imwrite('c.png', c)
    import pudb; pu.db

