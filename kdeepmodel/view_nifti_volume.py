#!/usr/bin/env python


import nibabel as nib
import numpy as np
import pyvista as pv


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='View Nifti volume')
    parser.add_argument('volume', type=str, help='input volume file')

    args = parser.parse_args()
    nifti = nib.load(args.volume)
    array = np.array(nifti.dataobj)
    if array.ndim == 4:
        for idx in range(array.shape[3]):
            vol = pv.wrap(array[:, :, :, idx])
            vol.plot(volume=True)
    else:
        vol = pv.wrap(array[:, :, :])
        vol.plot(volume=True)
