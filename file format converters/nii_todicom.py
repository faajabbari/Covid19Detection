import os
import SimpleITK as sitk
import numpy as np

nii_data_path = '/home/fateme/Desktop/ni/'
dicom_output_path = '/home/fateme/Desktop/ni/'
patients = os.listdir(nii_data_path)
patients.sort()

for s in patients:
    NiftyImg = sitk.ReadImage(nii_data_path + s)
    Img = sitk.GetArrayViewFromImage(NiftyImg)
    Slic_Num = len(Img)
    print(s, Slic_Num)
    for slic in range(Slic_Num):
        DicomFileName = dicom_output_path + s.split('.nii')[0] + "-" + str(slic) + ".dcm"
        img_npy = np.array(Img[slic,:,:])
        img_new = sitk.GetImageFromArray(img_npy)
        sitk.WriteImage(img_new, DicomFileName)
        print('Successifully write: ', slic)


# def export_segmentations_postprocess(indir, outdir):
#     # maybe_mkdir_p(outdir)
#     niftis = indir # subfiles(indir, suffix='nii.gz', join=False)
#     for n in niftis:
#         print("\n", n)
#         identifier = str(n.split("_")[-1][:-7])
#         outfname = os.path.join(outdir, "test-segmentation-%s.nii" % identifier)
#         img = sitk.ReadImage(os.path.join(indir, n))
#         img_npy = sitk.GetArrayFromImage(img)
#         lmap, num_objects = label((img_npy > 0).astype(int))
#         sizes = []
#         for o in range(1, num_objects + 1):
#             sizes.append((lmap == o).sum())
#         mx = np.argmax(sizes) + 1
#         print(sizes)
#         img_npy[lmap != mx] = 0
#         img_new = sitk.GetImageFromArray(img_npy)
#         img_new.CopyInformation(img)
#         sitk.WriteImage(img_new, outfname)
#
# indir= '/home/fateme/Desktop/ni/'
# outdir = '/home/fateme/Desktop/ni/'
# export_segmentations_postprocess(indir, outdir)