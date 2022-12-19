import os
import glob
import argparse

from medpy.io import load
import numpy as np
import png


def convert_to_png(file, save_dir):
    image_2d, image_header = load(os.path.join(file, file))
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    # Convert to uint8
    image_2d_scaled = np.uint8(image_2d_scaled)
    # Write the PNG file
    with open(os.path.join(save_dir, f'{file.strip(".dcm")}.png'), 'wb') as png_file:
        w = png.Writer(512, 512, greyscale=True)
        w.write(png_file, image_2d_scaled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', required=True,
                        help='address to dcm data')
    args = parser.parse_args()

    main_dir = args.main_dir
    save_dir = main_dir + '_png'
    os.makedirs(save_dir,  exist_ok=True)
    images = sorted(glob.glob(os.path.join(os.path.abspath(main_dir), '*.dcm')))
    for img in images:
        convert_to_png(img, save_dir)
