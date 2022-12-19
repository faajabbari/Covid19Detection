import os
import glob
import argparse

import numpy as np
import cv2
import scipy.ndimage


# applying filter on a single image
def apply_filter(filename, filter):
    print("reading file---> " + str(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = np.int64(img)
    h, w = img.shape
    print("shape: " + str(h) + " x " + str(w) + "\n")
    # define filters
    horizontal = filter
    vertical = np.transpose(filter)
    horizontalGrad = scipy.ndimage.convolve(img, horizontal, output=None,
                                            mode='reflect', cval=0.0, origin=0)
    verticalGrad = scipy.ndimage.convolve(img, vertical, output=None,
                                          mode='reflect', cval=0.0, origin=0)
    newgradientImage = np.sqrt(pow(abs(verticalGrad), 2.0) + \
                               pow(abs(horizontalGrad), 2.0))
    edges = newgradientImage
    # print(np.max(edges))
    # print(np.min(edges))

    return edges


# function for creating all edge-images of a directory
def convert_edge_dir(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        # applying Prewitt filter
        # for appyling any other filter change filter value accordingly
        # i.e. the 2nd args for apply_filter()
        imagemat = apply_filter(filename, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))

        ## do
        cv2.imwrite(destdir + '/img' + str(filecnt) + '.png',
                    imagemat)  # create the edge image and store
        # it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")


# function for creating all edge-images under both covid and non-covid directories
# since 2-class so COVID and NON-COVID present
def convert_edge_all_dir(coviddir, ncoviddir, destdir):
    # os.makedirs(os.path.join(destdir, '/COVID'), exist_ok=True)
    # os.makedirs(os.path.join(destdir, '/NON_COVID'), exist_ok=True)
    convert_edge_dir(coviddir, destdir + '/COVID')
    convert_edge_dir(ncoviddir, destdir + '/NON_COVID')
    print("\n---edge detection completed--\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coviddir', required=True,
                        help='address to covid directory')
    parser.add_argument('--ncoviddir', required=True,
                        help='address to non covid directory')
    parser.add_argument('--destdir', required=True,
                        help='address to directory to save edges')
    args = parser.parse_args()

    coviddir = args.coviddir  # Ex: 'data_subset/covid'
    ncoviddir = args.ncoviddir  # Ex: 'data_subset/normal'
    destdir = args.destdir  # Ex: edge_data_subset

    convert_edge_all_dir(coviddir, ncoviddir, destdir)
