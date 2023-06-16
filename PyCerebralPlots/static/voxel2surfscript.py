#!/usr/bin/env python

import os
import sys
import glob
import pygalmesh
import nibabel as nib
import numpy as np

from PyCerebralPlots.functions import get_neuroimaging_resources, convert_voxel, visualize_volume_iso_surface, visualize_volume_to_surface, visualize_freesurfer_annotation, visualize_surface_with_scalar_data, convert_fs, vectorized_surface_smooth

surfaces = np.sort(glob.glob("??_*.srf"))
for outname in surfaces:
	os.system("mv %s %s.backup" % (outname, outname))
	roi_index = int(outname[:2])
	aseg = nib.load('/applications/freesurfer/subjects/fsaverage/mri/aseg.mgz')
	data = aseg.get_fdata()
	affine = aseg.affine
	roi = np.zeros_like(data)
	roi[data==roi_index] = 1.
	vol = roi.astype(np.uint8)
	voxel_size = (1., 1., 1.)
	mesh = pygalmesh.generate_from_array(vol, voxel_size, max_facet_distance=0.5, max_cell_circumradius=1.0)
	faces = np.array(mesh.cells_dict['triangle'])
	vertices = np.array(mesh.points)
	vertices_affine_transformed = nib.affines.apply_affine(affine, vertices)
	nib.freesurfer.io.write_geometry(outname, vertices_affine_transformed, faces)
	os.system("mris_smooth -nw %s %s" % (outname, outname))

