#!/usr/bin/env python

#    Various functions for PyCerebralPlots
#    Copyright (C) 2023  Tristram Lett

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import sys
import imageio.v2 as imageio
import numpy as np
import warnings
import matplotlib.cbook

import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.colorbar import ColorbarBase
from scipy import ndimage
from scipy.special import erf
from skimage import filters, measure
from mayavi import mlab
from warnings import warn

def check_byteorder(arr):
	"""
	This function checks and ensures that the byte order (endianess) of the NumPy array matches the system's byte order.
	If the byte order does not match or is undefined, it performs the necessary byte swapping and changes the byte order of the array accordingly.

	Parameters
	----------
	arr : array-like
		The input neuroimage array whose byte order needs to be checked and adjusted if necessary.

	Returns
	-------
	np.ndarray
		The modified NumPy array with the correct byte order, either the same as the system's byte order or swapped if needed.
	"""
	arr = np.array(arr)
	if sys.byteorder == 'little':
		sys_bo = '<'
	elif sys.byteorder == 'big':
		sys_bo = '>'
	else:
		pass
	if not (arr.dtype.byteorder == sys_bo) or (arr.dtype.byteorder == '='):
		arr = arr.byteswap().newbyteorder()
	return(arr)

def _plot_colormap(cmap):
	data = np.arange(100).reshape((10, 10))  # Example data for the plot
	fig, ax = plt.subplots()
	im = ax.imshow(data, cmap=cmap)
	fig.colorbar(im)
	plt.show()

def create_rywlbb_gradient_cmap(linear_alpha = False, return_array = True):
	colors = ["#00008C", "#2234A8", "#4467C4", "#659BDF", "#87CEFB", "white", "#ffec19", "#ffc100", "#ff9800", "#ff5607", "#f6412d"]
	cmap = LinearSegmentedColormap.from_list("rywlbb_gradient", colors)
	cmap._init()  # Initialize the colormap
	if return_array:
		crange = np.linspace(0, 1, 256)
		cmap_array = cmap(crange)
		if linear_alpha:
			cmap_array[:,-1] = np.abs(np.linspace(-1, 1, 256))
		cmap_array *= 255
		cmap_array = cmap_array.astype(int)
		return(cmap_array)
	else:
		if linear_alpha:
			cmap._lut[:256, -1] = np.abs(np.linspace(-1, 1, 256))
		return(cmap)

def create_ryw_gradient_cmap(linear_alpha = False, return_array = True):
	colors = ["white", "#ffec19", "#ffc100", "#ff9800", "#ff5607", "#f6412d"]
	cmap = LinearSegmentedColormap.from_list("ryw_gradient", colors)
	cmap._init()  # Initialize the colormap
	if return_array:
		crange = np.linspace(0, 1, 256)
		cmap_array = cmap(crange)
		if linear_alpha:
			cmap_array[:,-1] = np.linspace(0, 1, 256)
		cmap_array *= 255
		cmap_array = cmap_array.astype(int)
		return(cmap_array)
	else:
		if linear_alpha:
			cmap._lut[:256, -1] = np.linspace(0, 1, 256)
		return(cmap)

def create_lbb_gradient_cmap(linear_alpha = False, return_array = True):
	colors = ["white", "#87CEFB", "#659BDF", "#4467C4", "#2234A8", "#00008C"]
	cmap = LinearSegmentedColormap.from_list("lbb_gradient", colors)
	cmap._init()  # Initialize the colormap
	if return_array:
		crange = np.linspace(0, 1, 256)
		cmap_array = cmap(crange)
		if linear_alpha:
			cmap_array[:,-1] = np.linspace(0, 1, 256)
		cmap_array *= 255
		cmap_array = cmap_array.astype(int)
		return(cmap_array)
	else:
		if linear_alpha:
			cmap._lut[:256, -1] = np.linspace(0, 1, 256)
		return(cmap)

def offscreen_render():
	mlab.options.offscreen = True

def convert_fs(fs_surface):
	v, f = nib.freesurfer.read_geometry(fs_surface)
	return(v, f)

def convert_voxel(img_data, affine = None, threshold = None, data_mask = None, absthreshold = None):
	"""
	Converts a voxel image to a surface including outputs voxel values to paint vertex surface.
	
	Parameters
	----------
	img_data : array
		image array
	affine : array
		 affine [4x4] to convert vertices values to native space (Default = None)
	data_mask : array
		use a mask to create a surface backbone (Default = None)
	threshold : float
		threshold for output of voxels (Default = None)
	absthreshold : float
		threshold for output of abs(voxels) (Default = None)
		
	Returns
	-------
		v : array
			vertices
		f : array
			faces
		values : array
			scalar values
	
	"""
	if threshold is not None:
		print("Zeroing data less than threshold = %1.2f" % threshold)
		img_data[img_data<threshold] = 0
	if absthreshold is not None:
		print("Zeroing absolute values less than threshold = %1.2f" % absthreshold)
		img_data[np.abs(img_data)<absthreshold] = 0
	if data_mask is not None:
		img_data[data_mask == 0] = 0
	try:
		v, f, _, values = measure.marching_cubes(img_data)
		if affine is not None:
			print("Applying affine transformation")
			v = nib.affines.apply_affine(affine,v)
	except:
		print("No voxels above threshold")
		v = f = values = None
	return(v, f, values)



def create_surface_adjacency(vertices, faces):
	adjacency = [set([]) for i in range(vertices.shape[0])]
	for i in range(faces.shape[0]):
		adjacency[faces[i, 0]].add(faces[i, 1])
		adjacency[faces[i, 0]].add(faces[i, 2])
		adjacency[faces[i, 1]].add(faces[i, 0])
		adjacency[faces[i, 1]].add(faces[i, 2])
		adjacency[faces[i, 2]].add(faces[i, 0])
		adjacency[faces[i, 2]].add(faces[i, 1])
	return(adjacency)

def vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = 5, scalar = None, lambda_w = 0.5, mode = 'laplacian', weighted = True):
	"""
	Applies Laplacian (Gaussian) or Taubin (low-pass) smoothing with option to smooth single volume
	
	Citations
	----------
	
	Herrmann, Leonard R. (1976), "Laplacian-isoparametric grid generation scheme", Journal of the Engineering Mechanics Division, 102 (5): 749-756.
	Taubin, Gabriel. "A signal processing approach to fair surface design." Proceedings of the 22nd annual conference on Computer graphics and interactive techniques. ACM, 1995.
	
	
	Parameters
	----------
	v : array
		vertex array
	f : array
		face array
	adjacency : array
		adjacency array

	
	Flags
	----------
	number_of_iter : int
		number of smoothing iterations
	scalar : array
		apply the same smoothing to a image scalar
	lambda_w : float
		lamda weighting of degree of movement for each iteration
		The weighting should never be above 1.0
	mode : string
		The type of smoothing can either be laplacian (which cause surface shrinkage) or taubin (no shrinkage)
		
	Returns
	-------
	v_smooth : array
		smoothed vertices array
	f : array
		f = face array (unchanged)
	
	Optional returns
	-------
	values : array
		smoothed scalar array
	
	"""
	assert mode in ['laplacian', 'taubin']
	
	if adjacency is None:
		adjacency = create_surface_adjacency(v, f)
	
	k = 0.1
	mu_w = -lambda_w/(1-k*lambda_w)

	lengths = np.array([len(a) for a in adjacency])
	maxlen = max(lengths)
	padded = [list(a) + [-1] * (maxlen - len(a)) for a in adjacency]
	adj = np.array(padded)
	w = np.ones(adj.shape, dtype=float)
	w[adj<0] = 0.
	val = (adj>=0).sum(-1).reshape(-1, 1)
	w /= val
	w = w.reshape(adj.shape[0], adj.shape[1],1)

	vorig = np.zeros_like(v)
	vorig[:] = v
	if scalar is not None:
		scalar[np.isnan(scalar)] = 0
		sorig = np.zeros_like(scalar)
		sorig[:] = scalar

	for iter_num in range(number_of_iter):
		if weighted:
			vadj = v[adj]
			vadj = np.swapaxes(v[adj],1,2)
			weights = np.zeros((v.shape[0], maxlen))
			for col in range(maxlen):
				weights[:,col] = np.power(np.linalg.norm(vadj[:,:,col] - v, axis=1),-1)
			weights[adj==-1] = 0
			vectors = np.einsum('abc,adc->acd', weights[:,None], vadj)
			if scalar is not None:
				scalar[np.isnan(scalar)] = 0
				sadj = scalar[adj]
				sadj[adj==-1] = 0
				if lambda_w < 1:
					scalar = (scalar*(1-lambda_w)) + lambda_w*(np.sum(np.multiply(weights, sadj),axis=1) / np.sum(weights, axis = 1))
				else:
					scalar = np.sum(np.multiply(weights, sadj),axis=1) / np.sum(weights, axis = 1)
				scalar[np.isnan(scalar)] = sorig[np.isnan(scalar)] # hacky scalar nan fix
			if iter_num % 2 == 0:
				v += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			elif mode == 'taubin':
				v += mu_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			elif mode == 'laplacian':
				v += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			else:
				print("Error: mode %s not understood" % mode)
				quit()
			v[np.isnan(v)] = vorig[np.isnan(v)] # hacky vertex nan fix
		else:
			if scalar is not None:
				sadj = scalar[adj]
				sadj[adj==-1] = 0

				if lambda_w < 1:
					scalar = (scalar*(1-lambda_w)) + (lambda_w*np.divide(np.sum(sadj, axis = 1),lengths))
				else:
					scalar = np.divide(np.sum(sadj, axis = 1),lengths)
			if iter_num % 2 == 0:
				v += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			elif mode == 'taubin':
				v += np.array(mu_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			elif mode == 'laplacian':
				v += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			else:
				pass
	if scalar is not None:
		return (v, f, scalar)
	else:
		return (v, f)

def plot_freesurfer_annotation_wireframe(v, f, freesurfer_annotation_path):
	labels, _, _ = nib.freesurfer.read_annot(freesurfer_annotation_path)
	a = [len(set(labels[f[k]])) != 1 for k in range(len(f))]
	scalar_out = np.zeros_like(labels).astype(np.float32)
	scalar_out[np.unique(f[a])] = 1
	surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
		scalars = scalar_out, 
		vmin = 0.5,
		vmax = 1,
		opacity = 0.6)
	sc_cmap_array = np.ones((256,4))*50
	sc_cmap_array[:,3] = 255
	sc_cmap_array[0] = [255,255,255,0]
	surf.actor.mapper.interpolate_scalars_before_mapping = 1
	surf.actor.property.backface_culling = True
	surf.module_manager.scalar_lut_manager.lut.table = sc_cmap_array
	surf.actor.actor.force_opaque = True

def create_annot_legend(labels, rgb_values, output_basename, num_columns = None, output_format = 'png', ratio = 4):
	num_boxes = len(labels)
	if num_columns is None:
		num_columns = int(np.sqrt(num_boxes / ratio))
	num_rows = np.ceil(num_boxes / num_columns)
	fig, ax = plt.subplots()
	ax.set_axis_off()
	handles = []
	colors = [tuple(val / 255 for val in rgb) for rgb in rgb_values]  # Normalize RGB values to range 0-1
	for color, label in zip(colors, labels):
		handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
	ax.legend(handles, labels, loc='center', ncol=num_columns, frameon=False)
	plt.xlim(0, num_columns)
	plt.ylim(0, num_rows)
	plt.tight_layout()
	plt.savefig("%s_annot_legend.%s" % (output_basename, output_format), transparent = True)
	plt.close()

def visualize_freesurfer_annotation(surface_path, freesurfer_annotation_path, atlas_values = None, cmap_array = None, add_wireframe = True, uniform_lighting = True, vmin = None, vmax = None, autothreshold_scalar = False, autothreshold_alg= 'yen_abs', absmin = None, absminmax = False, niter_surface_smooth = 0, save_figure = None, save_figure_orientation = 'x', output_format = 'png', output_transparent_background = True, color_bar_label = None):
	if save_figure is not None:
		mlab.options.offscreen = True
	labels, ctab, names = nib.freesurfer.read_annot(freesurfer_annotation_path)
	roi_indices = np.unique(labels)[1:]
	v, f = convert_fs(surface_path)
	if niter_surface_smooth > 0:
		v, f = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth)
	if atlas_values is None:
		roi_indices = np.unique(labels)
		colors = ctab[roi_indices][:,:3]
		cindices = np.arange(0,len(colors),1)
		scalar_data = np.zeros((len(labels)))
		for r, roi in enumerate(roi_indices):
			scalar_data[labels==roi] = cindices[r]
		# create a new cmap array
		cmap_array = np.ones((256,4), dtype=int) * 255
		cmap_array[:colors.shape[0],:3] = colors
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = scalar_data, 
			vmin = 0,
			vmax = 255)
		surf.module_manager.scalar_lut_manager.lut.table = cmap_array
		surf.actor.mapper.interpolate_scalars_before_mapping = 0
	else:
		assert cmap_array is not None, "Error: a cmap_array must be provided for plotting atlas_values"
		scalar_data = np.zeros((len(labels)))
		for r, roi in enumerate(roi_indices):
			scalar_data[labels==roi] = atlas_values[r]
		if autothreshold_scalar:
			vmin, vmax = perform_autothreshold(scalar_data, threshold_type = autothreshold_alg)
		if absmin is not None:
			scalar_data[np.abs(scalar_data) < absmin] = 0
		if vmin is None:
			vmin = np.nanmin(scalar_data)
		if vmax is None:
			vmax = np.nanmax(scalar_data)
		if absminmax:
			vmax = np.max([np.abs(vmin), np.abs(vmax)])
			vmin = -np.max([np.abs(vmin), np.abs(vmax)])
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = scalar_data, 
			vmin = vmin,
			vmax = vmax)
	if add_wireframe:
		plot_freesurfer_annotation_wireframe(v, f, freesurfer_annotation_path)
	surf.module_manager.scalar_lut_manager.lut.table = cmap_array
	surf.actor.mapper.interpolate_scalars_before_mapping = 0
	surf.actor.property.backface_culling = True
	surf.scene.parallel_projection = True
	surf.scene.background = (0,0,0)
	surf.scene.x_minus_view()
	if uniform_lighting:
		surf.actor.property.lighting = False
	if save_figure is not None:
		if 'x' in save_figure_orientation:
			savename = '%s_left.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.x_plus_view()
			savename = '%s_right.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'y' in save_figure_orientation:
			surf.scene.y_minus_view()
			savename = '%s_posterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 270, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.y_plus_view()
			savename = '%s_anterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 90, crop_black=True, b_transparent=output_transparent_background)
		if 'z' in save_figure_orientation:
			surf.scene.z_minus_view()
			savename = '%s_inferior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.z_plus_view()
			savename = '%s_superior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'iso' in save_figure_orientation:
			surf.scene.isometric_view()
			savename = '%s_isometric.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if atlas_values is not None:
			write_colorbar(output_basename = save_figure,
								cmap_array = cmap_array,
								vmax = vmax,
								vmin = vmin,
								colorbar_label = None,
								output_format = 'png',
								abs_colorbar = False,
								n_ticks = 11,
								orientation='vertical')
		else:
			create_annot_legend(labels = names, rgb_values = ctab[roi_indices][:,:3], output_basename = save_figure)
		mlab.clf()
		mlab.options.offscreen = False

def plot_freesurfer_annotation_wireframe2(v, f, freesurfer_annotation_path):
	labels, _, _ = nib.freesurfer.read_annot(freesurfer_annotation_path)
	a = [len(set(labels[f[k]])) != 1 for k in range(len(f))]
	scalar_out = np.zeros_like(labels).astype(np.float32)
	scalar_out[np.unique(f[a])] = 1
	surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
		scalars = scalar_out, 
		vmin = 0.5,
		vmax = 1,
		opacity = 0.6)
	sc_cmap_array = np.ones((256,4))*50
	sc_cmap_array[:,3] = 255
	sc_cmap_array[0] = [255,255,255,0]
	surf.actor.mapper.interpolate_scalars_before_mapping = 1
	surf.actor.property.backface_culling = True
	surf.module_manager.scalar_lut_manager.lut.table = sc_cmap_array
	surf.actor.actor.force_opaque = True

def autothreshold_both_hemispheres_mgh(lh_mgh_image, rh_mgh_image, autothreshold_alg = 'yen_abs'):
	"""
	Applies automatic thresholding to both hemispheres of an MGH image.

	This function loads the MGH images for both the left and right hemispheres,
	concatenates their data arrays, and performs automatic thresholding using the
	specified algorithm. The minimum and maximum threshold values are then returned.

	Example usage:
	lh_mgh_image = 'path/to/left_hemisphere.mgh'
	rh_mgh_image = 'path/to/right_hemisphere.mgh'
	vmin, vmax = autothreshold_both_hemispheres_mgh(lh_mgh_image, rh_mgh_image, autothreshold_alg='yen_abs')

	Parameters
	----------
		lh_mgh_image : str
			Path to the left hemisphere MGH image file.
		rh_mgh_image : str
			Path to the right hemisphere MGH image file.
		autothreshold_alg : str, optional
			Autothreshold algorithm to use. Default is 'yen_abs'.

	Returns
	-------
		vmin : float
			Minimum threshold value calculated using the specified autothreshold algorithm.
		vmax : float
			Maximum threshold value calculated using the specified autothreshold algorithm.
	"""
	lh_data = nib.load(lh_mgh_image).get_fdata()
	rh_data = nib.load(rh_mgh_image).get_fdata()
	if (lh_data.ndim == 4) or (rh_data.ndim == 4):
		warn("Multiple volumes detected. Thresholding will be performed on all volumes.")
	data = np.concatenate((np.squeeze(lh_data), np.squeeze(rh_data)))
	vmin, vmax = perform_autothreshold(data, threshold_type = autothreshold_alg)
	return(vmin, vmax)

def visualize_surface_with_scalar_data(surface, mgh_image = None, cmap_array = None, vmin = None, vmax = None, autothreshold_scalar = False, autothreshold_alg= 'yen_abs', absmin = None, absminmax = False, transparent = False, save_figure = None, save_figure_orientation = 'x', output_format = 'png', output_transparent_background = True, color_bar_label = None, niter_surface_smooth = 0, render_annotation_wireframe_path = None, uniform_lighting = False):
	"""
	Renders a freesurfer surface with optional scalar data using Mayavi. The surface will be interactively rendered if the save_figure is None.

	Example usage:
	_ = visualize_surface_with_scalar_data(surface = os.environ['SUBJECTS_DIR'] + '/fsaverage/surf/lh.pial_semi_inflated',
																	mgh_image = "lh_statistic.mgh"
																	cmap_array = create_rywlbb_gradient_cmap()
																	autothreshold_scalar = True,
																	absminmax = True)

	Parameters
	----------
		surface : str
			The path to the freesurfer surface. e.g., surface = os.environ['SUBJECTS_DIR']+'/fsaverage/surf/lh.pial_semi_inflated'
		mgh_image : str, optional
			The path to a MGH image file. Defaults to None.
		cmap_array : array, optional
			The colormap array with shape (256,4) and int values ranging from 0 to 255. Defaults to None. e.g., cmap_array = create_rywlbb_gradient_cmap()
		vmin : float, optional
			The minimum value of the scalar data. Defaults to None.
		vmax : float, optional
			The maximum value of the scalar data. Defaults to None.
		autothreshold_scalar : bool, optional
			Whether to perform autothresholding on the scalar data. Defaults to False.
		autothreshold_alg : str, optional
			The autothresholding algorithm to use {yen, otso, li}. Defaults to 'yen_abs'.
			Note: adding _p thresholds only positive values, and _abs sign flips the scalar data so the mean non-zero value is positive. e.g., 'yen_abs'
		absmin : float, optional
			The absolute minimum threshold for the scalar data. Defaults to None.
		absminmax : bool, optional
			Whether to use the absolute minimum and maximum values for the colormap range.
			Defaults to False.
		transparent : bool, optional
			Whether the surface should be rendered with transparency. Defaults to False.
		save_figure : str, optional
			The basename string for saving the figures. If not None, an image and colorbar will saved using the basename and output_format. Defaults to None.
		save_figure_orientation : str, optional
			The orientations to save the figures. Defaults to 'x' (left and right views). Valid options are: {x, y, z, iso}. e.g., save_figure_orientation = 'xyz'
		output_format : str, optional
			The output file format for the saved figures. Defaults to 'png'.
		color_bar_label : str, optional
			The label for the color bar. Defaults to None.
		niter_surface_smooth : int, optional
			The number of iterations for surface smoothing. Defaults to 0. If mgh_image is not None, the scalar values will also be smoothed.
		render_annotation_wireframe_path : str, optional
			The path to the Freesurfer annotation. This option will add dark grey outline of the annotation as a wireframe. Defaults to None.
	Returns
	-------
		surf : object
			The mayavi scene instance
	"""
	if save_figure is not None:
		mlab.options.offscreen = True
	if cmap_array is None:
		cmap_array = np.ones((256,4),int) * 255
	v, f = convert_fs(surface)
	if mgh_image is  None:
		if niter_surface_smooth > 0:
			v, f, = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth, scalar = None)
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f, scalars = None)
	else:
		img = nib.load(mgh_image)
		invol = np.asanyarray(img.dataobj)
		scalar_data = check_byteorder(np.squeeze(invol))
		if niter_surface_smooth > 0:
			v, f, scalar_data = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth, scalar = scalar_data)
		if autothreshold_scalar:
			if mgh_image is not None:
				vmin, vmax = perform_autothreshold(scalar_data, threshold_type = autothreshold_alg)
		if absmin is not None:
			scalar_data[np.abs(scalar_data) < absmin] = 0
		if vmin is None:
			vmin = np.nanmin(invol)
		if vmax is None:
			vmax = np.nanmax(invol)
		if absminmax:
			vmax = np.max([np.abs(vmin), np.abs(vmax)])
			vmin = -np.max([np.abs(vmin), np.abs(vmax)])
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = scalar_data, 
			vmin = vmin,
			vmax = vmax,
			transparent = transparent)
	if render_annotation_wireframe_path is not None:
		plot_freesurfer_annotation_wireframe(v = v,
														f = f,
														freesurfer_annotation_path = render_annotation_wireframe_path)
	surf.module_manager.scalar_lut_manager.lut.table = cmap_array
	surf.actor.mapper.interpolate_scalars_before_mapping = 0
	surf.actor.property.backface_culling = True
	surf.scene.parallel_projection = True
	surf.scene.background = (0,0,0)
	surf.scene.x_minus_view()
	if uniform_lighting:
		surf.actor.property.lighting = False
	if save_figure is not None:
		if 'x' in save_figure_orientation:
			savename = '%s_left.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.x_plus_view()
			savename = '%s_right.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'y' in save_figure_orientation:
			surf.scene.y_minus_view()
			savename = '%s_posterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 270, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.y_plus_view()
			savename = '%s_anterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 90, crop_black=True, b_transparent=output_transparent_background)
		if 'z' in save_figure_orientation:
			surf.scene.z_minus_view()
			savename = '%s_inferior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.z_plus_view()
			savename = '%s_superior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'iso' in save_figure_orientation:
			surf.scene.isometric_view()
			savename = '%s_isometric.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		write_colorbar(output_basename = save_figure,
							cmap_array = cmap_array,
							vmax = vmax,
							vmin = vmin,
							colorbar_label = None,
							output_format = 'png',
							abs_colorbar = False,
							n_ticks = 11,
							orientation='vertical')
		mlab.clf()
		mlab.options.offscreen = False
	return(surf)

# os.environ['FSLDIR']
#nifti_image_path = '/mnt/raid1/projects/tris/RESULTS_WRITEUP_SCCA/08MAY2023/EmotionalFaceTask_loading_comp3_tfce_rs.nii.gz'
#render_mask_volume = '/mnt/raid1/projects/tris/RESULTS_WRITEUP_SCCA/08MAY2023/NEWSURF/MNI152_T1_1mm_brain.nii.gz'
#visualize_volume_to_surface(nifti_image_path = 'EmotionalFaceTask_loading_comp3_tfce_rs.nii.gz', cmap_array = create_rywlbb_gradient_cmap(), absminmax = True, render_mask_volume = 'NEWSURF/MNI152_T1_1mm_brain.nii.gz', save_figure = 'EmotionalFaceTask_loading_comp3_tfce_rs',save_figure_orientation = 'xz', autothreshold_scalar = True, volume_opacity = 0.8, niter_surface_smooth = 0, niter_surface_smooth_render_mask = 8)

def visualize_volume_iso_surface(nifti_image_path, cmap_array, nifti_image_mask = None,  vmin = None, vmax = None, volume_opacity = 0.8, n_contours = 11, autothreshold_scalar = False, autothreshold_alg= 'yen_abs', absmin = None, absminmax = False, render_mask_volume = None, binarize_render_mask_volume = True, render_mask_volume_opacity = 0.8, save_figure = None, save_figure_orientation = 'xz', output_format = 'png', output_transparent_background = True, color_bar_label = None, niter_surface_smooth = 0, niter_surface_smooth_render_mask = 0):
	"""
	Visualizes a 3D volume as an iso-surface using Mayavi's mlab module.

	Parameters
	----------
		nifti_image_path : str
			Path to the NIfTI image file.
		cmap_array : numpy.ndarray
			Colormap array for the iso-surface visualization.
		nifti_image_mask : str, optional
			Path to the NIfTI mask file. (default: None)
		vmin : float, optional
			Minimum value for the scalar data. (default: None)
		vmax : float, optional
			Maximum value for the scalar data. (default: None)
		volume_opacity : float, optional
			Opacity of the volume visualization. (default: 0.8)
		n_contours : int, optional
			Number of contours to display
		autothreshold_scalar : bool, optional
			Flag to enable automatic thresholding of the scalar data. (default: False)
		autothreshold_alg : str, optional
			Algorithm for automatic thresholding. (default: 'yen_abs')
		absmin : float, optional
			Threshold value for absolute minimum of the scalar data. (default: None)
		absminmax : bool, optional
			Flag to adjust vmin and vmax based on the absolute minimum and maximum values. (default: False)
		render_mask_volume : str, optional
			Path to the NIfTI image file for rendering a separate mask volume. (default: None)
		binarize_render_mask_volume : bool, optional
			Flag to binarize the render mask volume. (default: True)
		render_mask_volume_opacity : float, optional
			Opacity of the render mask volume visualization. (default: 0.8)
		save_figure : str, optional
			Path to save the figure. (default: None)
		save_figure_orientation : str, optional
			Orientation of the saved figure (e.g., 'xz', 'xy', 'yz', 'iso'). (default: 'xz')
		output_format : str, optional
			Format of the saved figure. (default: 'png')
		output_transparent_background : bool, optional
			Flag to make the saved figure's background transparent. (default: True)
		color_bar_label : str, optional
			Label for the colorbar. (default: None)
		niter_surface_smooth : int, optional
			Number of iterations for smoothing the iso-surface. (default: 0)
		niter_surface_smooth_render_mask : int, optional
			Number of iterations for smoothing the render mask volume. (default: 0)
	Returns
	-------
		None
	"""

	if save_figure is not None:
		mlab.options.offscreen = True
	invol = nib.as_closest_canonical(nib.load(nifti_image_path))
	data = check_byteorder(np.asanyarray(invol.dataobj))
	if nifti_image_mask is None:
		mask_arr = data!=0
	else:
		mask = nib.as_closest_canonical(nib.load(nifti_image_mask))
		mask_arr = check_byteorder(np.asanyarray(mask.dataobj)) != 0
	scalar_data = data[mask_arr]
	if autothreshold_scalar:
		vmin, vmax = perform_autothreshold(scalar_data, threshold_type = autothreshold_alg)
	if absmin is not None:
		scalar_data[np.abs(scalar_data) < absmin] = 0
	if vmin is None:
		vmin = np.nanmin(scalar_data)
	if vmax is None:
		vmax = np.nanmax(scalar_data)
	if absminmax:
		vmax = np.max([np.abs(vmin), np.abs(vmax)])
		vmin = -np.max([np.abs(vmin), np.abs(vmax)])
	src = apply_affine_to_scalar_field(data, affine = invol.affine)
	surf = mlab.pipeline.iso_surface(src,
												vmin=vmin,
												vmax=vmax,
												opacity = volume_opacity,
												contours = n_contours)
	surf.module_manager.scalar_lut_manager.lut.table = cmap_array
	surf.actor.mapper.interpolate_scalars_before_mapping = 1
	surf.module_manager.scalar_lut_manager.lut.table = cmap_array
	surf.actor.mapper.interpolate_scalars_before_mapping = 1
	if render_mask_volume is not None:
		render_mask = nib.as_closest_canonical(nib.load(render_mask_volume))
		mask_data = check_byteorder(np.asanyarray(render_mask.dataobj))
		if binarize_render_mask_volume:
			mask_data[mask_data!=0] = 1
		v, f, scalar_mask_data = convert_voxel(mask_data, affine = render_mask.affine)
		if niter_surface_smooth_render_mask > 0:
			v, f = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth_render_mask, scalar = None)
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = scalar_mask_data, 
			vmin = .99,
			vmax = 1.01,
			opacity = render_mask_volume_opacity)
		white_cmap_arr = np.ones((256, 4)) * 200
		white_cmap_arr[:,3] = int(render_mask_volume_opacity * 255)
		surf.module_manager.scalar_lut_manager.lut.table = white_cmap_arr
	surf.actor.mapper.interpolate_scalars_before_mapping = 0
	surf.actor.property.backface_culling = True
	surf.scene.parallel_projection = True
	surf.scene.background = (0,0,0)
	surf.scene.x_minus_view()
	if save_figure is not None:
		if 'x' in save_figure_orientation:
			savename = '%s_left.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.x_plus_view()
			savename = '%s_right.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'y' in save_figure_orientation:
			surf.scene.y_minus_view()
			savename = '%s_posterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 270, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.y_plus_view()
			savename = '%s_anterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 90, crop_black=True, b_transparent=output_transparent_background)
		if 'z' in save_figure_orientation:
			surf.scene.z_minus_view()
			savename = '%s_superior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.z_plus_view()
			savename = '%s_inferior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'iso' in save_figure_orientation:
			surf.scene.isometric_view()
			savename = '%s_isometric.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		write_colorbar(output_basename = save_figure,
							cmap_array = cmap_array,
							vmax = vmax,
							vmin = vmin,
							colorbar_label = None,
							output_format = 'png',
							abs_colorbar = False,
							n_ticks = 11,
							orientation='vertical')
		mlab.clf()
		mlab.options.offscreen = False
	return(surf)

def visualize_volume_to_surface(nifti_image_path, cmap_array, nifti_image_mask = None,  volume_opacity = 0.8, vmin = None, vmax = None, autothreshold_scalar = False, autothreshold_alg= 'yen_abs', absmin = None, absminmax = False, render_mask_volume = None, binarize_render_mask_volume = True, render_mask_volume_opacity = 0.8, save_figure = None, save_figure_orientation = 'xz', output_format = 'png', output_transparent_background = True, color_bar_label = None, niter_surface_smooth = 0, niter_surface_smooth_render_mask = 0):
	"""
	Visualizes a 3D volume as a surface using Mayavi's mlab module.

	Parameters
	----------
		nifti_image_path : str
			Path to the NIfTI image file.
		cmap_array : numpy.ndarray
			Colormap array for the surface visualization.
		nifti_image_mask : str, optional
			Path to the NIfTI mask file. (default: None)
		volume_opacity : float, optional
			Opacity of the volume visualization. (default: 0.8)
		vmin : float, optional
			Minimum value for the scalar data. (default: None)
		vmax : float, optional
			Maximum value for the scalar data. (default: None)
		autothreshold_scalar : bool, optional
			Flag to enable automatic thresholding of the scalar data. (default: False)
		autothreshold_alg : str, optional
			Algorithm for automatic thresholding. (default: 'yen_abs')
		absmin : float, optional
			Threshold value for absolute minimum of the scalar data. (default: None)
		absminmax : bool, optional
			Flag to adjust vmin and vmax based on the absolute minimum and maximum values. (default: False)
		render_mask_volume : str, optional
			Path to the NIfTI image file for rendering a separate mask volume. (default: None)
		binarize_render_mask_volume : bool, optional
			Flag to binarize the render mask volume. (default: True)
		render_mask_volume_opacity : float, optional
			Opacity of the render mask volume visualization. (default: 0.8)
		save_figure : str, optional
			Path to save the figure. (default: None)
		save_figure_orientation : str, optional
			Orientation of the saved figure (e.g., 'x', 'xz', 'xy', 'yz', 'iso'). (default: 'xz')
		output_format : str, optional
			Format of the saved figure. (default: 'png')
		output_transparent_background : bool, optional
			Flag to make the saved figure's background transparent. (default: True)
		color_bar_label : str, optional
			Label for the colorbar. (default: None)
		niter_surface_smooth : int, optional
			Number of iterations for smoothing the surface. (default: 0)
		niter_surface_smooth_render_mask : int, optional
			Number of iterations for smoothing the render mask volume. (default: 0)
	Returns
	-------
		None
	"""
	if save_figure is not None:
		mlab.options.offscreen = True
	invol = nib.as_closest_canonical(nib.load(nifti_image_path))
	data = check_byteorder(np.asanyarray(invol.dataobj))
	if nifti_image_mask is None:
		mask_arr = data!=0
	else:
		mask = nib.as_closest_canonical(nib.load(nifti_image_mask))
		mask_arr = check_byteorder(np.asanyarray(mask.dataobj)) != 0
	scalar_data = data[mask_arr]
	if autothreshold_scalar:
		vmin, vmax = perform_autothreshold(scalar_data, threshold_type = autothreshold_alg)
	if absmin is not None:
		scalar_data[np.abs(scalar_data) < absmin] = 0
	if vmin is None:
		vmin = np.nanmin(scalar_data)
	if vmax is None:
		vmax = np.nanmax(scalar_data)
	if absminmax:
		vmax = np.max([np.abs(vmin), np.abs(vmax)])
		vmin = -np.max([np.abs(vmin), np.abs(vmax)])
	if np.max(data) > 0:
		# positive data
		data_pos = np.array(data)
		data_pos[data_pos<0] = 0
		v, f, scalar_data = convert_voxel(data_pos, affine = invol.affine)
		if niter_surface_smooth > 0:
			v, f, scalar_data = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth, scalar = scalar_data)
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = scalar_data, 
			vmin = vmin,
			vmax = vmax,
			opacity = volume_opacity)
	if np.min(data) < 0:
		data_neg = np.array(data)
		data_neg[data_neg>0] = 0
		v, f, scalar_data = convert_voxel(data_neg*-1, affine = invol.affine)
		if niter_surface_smooth > 0:
			v, f, scalar_data = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth, scalar = scalar_data)
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = -scalar_data, 
			vmin = vmin,
			vmax = vmax,
			opacity = volume_opacity)
	surf.module_manager.scalar_lut_manager.lut.table = cmap_array
	surf.actor.mapper.interpolate_scalars_before_mapping = 1
	if render_mask_volume is not None:
		render_mask = nib.as_closest_canonical(nib.load(render_mask_volume))
		mask_data = check_byteorder(np.asanyarray(render_mask.dataobj))
		if binarize_render_mask_volume:
			mask_data[mask_data!=0] = 1
		v, f, scalar_mask_data = convert_voxel(mask_data, affine = render_mask.affine)
		if niter_surface_smooth_render_mask > 0:
			v, f = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth_render_mask, scalar = None)
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = scalar_mask_data, 
			vmin = .99,
			vmax = 1.01,
			opacity = render_mask_volume_opacity)
		white_cmap_arr = np.ones((256, 4)) * 200
		white_cmap_arr[:,3] = int(render_mask_volume_opacity * 255)
		surf.module_manager.scalar_lut_manager.lut.table = white_cmap_arr
	surf.actor.mapper.interpolate_scalars_before_mapping = 0
	surf.actor.property.backface_culling = True
	surf.scene.parallel_projection = True
	surf.scene.background = (0,0,0)
	surf.scene.x_minus_view()
	if save_figure is not None:
		if 'x' in save_figure_orientation:
			savename = '%s_left.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.x_plus_view()
			savename = '%s_right.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'y' in save_figure_orientation:
			surf.scene.y_minus_view()
			savename = '%s_posterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 270, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.y_plus_view()
			savename = '%s_anterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 90, crop_black=True, b_transparent=output_transparent_background)
		if 'z' in save_figure_orientation:
			surf.scene.z_minus_view()
			savename = '%s_superior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			surf.scene.z_plus_view()
			savename = '%s_inferior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if 'iso' in save_figure_orientation:
			surf.scene.isometric_view()
			savename = '%s_isometric.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		write_colorbar(output_basename = save_figure,
							cmap_array = cmap_array,
							vmax = vmax,
							vmin = vmin,
							colorbar_label = None,
							output_format = 'png',
							abs_colorbar = False,
							n_ticks = 11,
							orientation='vertical')
		mlab.clf()
		mlab.options.offscreen = False
	return(surf)

def visualize_volume_contour_with_scalar_data(nifti_image_path = None, nifti_image_mask = None, cmap_array = None, volume_opacity = 0.8, n_contours = 20, vmin = None, vmax = None, autothreshold_scalar = False, autothreshold_alg= 'yen_abs', absmin = None, absminmax = False, render_mask_volume = None, save_figure = None, save_figure_orientation = 'xz', output_format = 'png', color_bar_label = None, niter_surface_smooth = 10):
	if save_figure is not None:
		mlab.options.offscreen = True
	invol = nib.as_closest_canonical(nib.load(nifti_image_path))
	data = check_byteorder(np.asanyarray(invol.dataobj))
	if nifti_image_mask is None:
		mask_arr = data!=0
	else:
		mask = nib.as_closest_canonical(nib.load(nifti_image_mask))
		mask_arr = check_byteorder(np.asanyarray(mask.dataobj)) != 0
	scalar_data = data[mask_arr]
	if autothreshold_scalar:
		vmin, vmax = perform_autothreshold(scalar_data, threshold_type = autothreshold_alg)
	if absmin is not None:
		scalar_data[np.abs(scalar_data) < absmin] = 0
	if vmin is None:
		vmin = np.nanmin(scalar_data)
	if vmax is None:
		vmax = np.nanmax(scalar_data)
	if absminmax:
		vmax = np.max([np.abs(vmin), np.abs(vmax)])
		vmin = -np.max([np.abs(vmin), np.abs(vmax)])
	try:
		surf = apply_affine_to_contour3d(data, invol.affine,
			lthresh = np.round(vmin, 3) + 0.001,
			hthresh = np.round(vmax, 3) - 0.001,
			contours = n_contours,
			opacity = volume_opacity)
	except:
		data[0,0,0] = np.round(vmin, 3) - 0.001
		data[-1,0,0] = np.round(vmax, 3) + 0.001
		surf = apply_affine_to_contour3d(data, invol.affine,
			lthresh = np.round(vmin, 3) + 0.001,
			hthresh = np.round(vmax, 3) - 0.001,
			contours = n_contours,
			opacity = volume_opacity)
	surf.contour.minimum_contour = vmin
	surf.module_manager.scalar_lut_manager.lut.table = cmap_array
	surf.actor.mapper.interpolate_scalars_before_mapping = 1
	if render_mask_volume is not None:
		render_mask = nib.as_closest_canonical(nib.load(render_mask_volume))
		mask_data = check_byteorder(np.asanyarray(render_mask.dataobj))
		mask_data[mask_data!=0] = 1
		v, f, scalar_mask_data = convert_voxel(mask_data, affine = render_mask.affine)
		if niter_surface_smooth > 0:
			v, f = vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = niter_surface_smooth, scalar = None)
		surf = mlab.triangular_mesh(v[:,0], v[:,1], v[:,2], f,
			scalars = scalar_mask_data, 
			vmin = .99,
			vmax = 1.01,
			opacity = volume_opacity)
		white_cmap_arr = np.ones((256, 4)) * 255
		white_cmap_arr[:,3] = int(volume_opacity * 255)
		surf.module_manager.scalar_lut_manager.lut.table = white_cmap_arr
	surf.actor.mapper.interpolate_scalars_before_mapping = 0
	surf.actor.property.backface_culling = True
	surf.scene.parallel_projection = True
	surf.scene.background = (0,0,0)
	surf.scene.x_minus_view()
	if save_figure is not None:
		if 'x' in save_figure_orientation:
			savename = '%s_left.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True)
			surf.scene.x_plus_view()
			savename = '%s_right.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True)
		if 'y' in save_figure_orientation:
			surf.scene.y_minus_view()
			savename = '%s_posterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 270, crop_black=True)
			surf.scene.y_plus_view()
			savename = '%s_anterior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, rotate = 90, crop_black=True)
		if 'z' in save_figure_orientation:
			surf.scene.z_minus_view()
			savename = '%s_inferior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True)
			surf.scene.z_plus_view()
			savename = '%s_superior.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True)
		if 'iso' in save_figure_orientation:
			surf.scene.isometric_view()
			savename = '%s_isometric.%s'  % (save_figure, output_format)
			mlab.savefig(savename, magnification=4)
			correct_image(savename, crop_black=True)
		write_colorbar(output_basename = save_figure,
							cmap_array = cmap_array,
							vmax = vmax,
							vmin = vmin,
							colorbar_label = None,
							output_format = 'png',
							abs_colorbar = False,
							n_ticks = 11,
							orientation='vertical')
		mlab.clf()
		mlab.options.offscreen = False
	return(surf)

def write_colorbar(output_basename, cmap_array, vmax, vmin = None, colorbar_label = None, output_format = 'png', abs_colorbar = False, n_ticks = 11, orientation='vertical'):
	"""
	Generate a colorbar and save it as an image file.

	Parameters
	----------
		output_basename: str 
			Base name of the output file.
		cmap_array: ndarray 
			Array of colors representing the colormap.
		vmax: float
			Maximum value for the colorbar.
		vmin: float, optional
			Minimum value for the colorbar. If not provided, it defaults to 0.
		colorbar_label: str, optional
			Label for the colorbar.
		output_format: str, optional
			Output file format. Defaults to 'png'.
		abs_colorbar: bool, optional
			Whether to treat the colorbar as absolute, making vmin = -vmax if True. Defaults to False.
		n_ticks: int, optional
			Number of ticks to be displayed on the colorbar. Defaults to 11.
		orientation: str, optional
			Orientation of the colorbar ('vertical' or 'horizontal'). Defaults to 'vertical'.
	Returns:
		None
	"""
	if abs_colorbar:
		vmin = -vmax
	if vmin is None:
		vmin = 0
	cmap = ListedColormap(np.divide(cmap_array,255))
	if orientation == 'horizontal':
		fig, ax = plt.subplots(figsize=(4, 1))
	else:
		fig, ax = plt.subplots(figsize=(1, 4))
	colorbar = ColorbarBase(ax, cmap=cmap, orientation=orientation)
	if colorbar_label is not None:
		colorbar.set_label(colorbar_label)
	tick_labels = ["%1.2f" % t for t in np.linspace(-vmax, vmax, n_ticks)]
	colorbar.set_ticks(ticks = np.linspace(0, 1, n_ticks), labels = tick_labels)
	plt.tight_layout()
	plt.savefig("%s_colorbar.%s" % (output_basename, output_format), transparent = True)
	plt.close()

def screenshot_scene(surf, output_basename, save_figure_orientation = 'x', output_format = 'png'):
	surf.scene.parallel_projection = True
	surf.scene.background = (0,0,0)
	surf.scene.x_minus_view()
	if 'x' in save_figure_orientation:
		savename = '%s_left.%s'  % (output_basename, output_format)
		mlab.savefig(savename, magnification=4)
		correct_image(savename)
		surf.scene.x_plus_view()
		savename = '%s_right.%s'  % (output_basename, output_format)
		mlab.savefig(savename, magnification=4)
		correct_image(savename)
	if 'y' in save_figure_orientation:
		surf.scene.y_minus_view()
		savename = '%s_posterior.%s'  % (output_basename, output_format)
		mlab.savefig(savename, magnification=4)
		correct_image(savename, rotate = 270, b_transparent = True)
		surf.scene.y_plus_view()
		savename = '%s_anterior.%s'  % (output_basename, output_format)
		mlab.savefig(savename, magnification=4)
		correct_image(savename, rotate = 90, b_transparent = True)
	if 'z' in save_figure_orientation:
		surf.scene.z_minus_view()
		savename = '%s_inferior.%s'  % (output_basename, output_format)
		mlab.savefig(savename, magnification=4)
		correct_image(savename)
		surf.scene.z_plus_view()
		savename = '%s_superior.%s'  % (output_basename, output_format)
		mlab.savefig(savename, magnification=4)
		correct_image(savename)
	if 'iso' in save_figure_orientation:
		surf.scene.isometric_view()
		savename = '%s_isometric.%s'  % (output_basename, output_format)
		mlab.savefig(savename, magnification=4)
		correct_image(savename)
	mlab.clf()


def apply_affine_to_scalar_field(data, affine):
	"""
	Applies the given affine transformation to the coordinates of the scalar field.

	Parameters
	----------
	data : array-like
		The input scalar field data.

	affine : array-like
		The affine transformation matrix to be applied.

	Returns
	-------
	src : mlab.pipeline.scalar_field
		The scalar field source with transformed coordinates.

	Notes
	-----
	- If the input data is 4D, only the first volume will be displayed.
	- The function assumes that the scalar field data has non-zero values at specific coordinates.

	"""
	data = np.array(data)
	if data.ndim == 4: # double check
		print("4D volume detected. Only the first volume will be displayed.")
		data = data[:,:,:,0]
	size_x, size_y, size_z = data.shape
	x,y,z = np.where(data!=55378008)
	coord = np.column_stack((x,y))
	coord = np.column_stack((coord,z))
	coord_array = nib.affines.apply_affine(affine, coord)
	xi = coord_array[:,0].reshape(size_x, size_y, size_z)
	yi = coord_array[:,1].reshape(size_x, size_y, size_z)
	zi = coord_array[:,2].reshape(size_x, size_y, size_z)
	src = mlab.pipeline.scalar_field(xi, yi, zi, data)
	return(src)

def apply_affine_to_contour3d(data, affine, lthresh, hthresh, name = "", contours=15, opacity=0.7):
	"""
	Applies the given affine transformation to the coordinates of the 3D contour data.

	Parameters
	----------
	data : array-like
		The input 3D contour data.

	affine : array-like
		The affine transformation matrix to be applied.

	lthresh : float
		The lower threshold value for the contours.

	hthresh : float
		The upper threshold value for the contours.

	name : str
		The name of the contour visualization.

	contours : int, optional
		The number of contours to be generated (default is 15).

	opacity : float, optional
		The opacity of the contours (default is 0.7).

	Returns
	-------
	src : mlab.pipeline.surface
		The contour surface with transformed coordinates.

	Notes
	-----
	- If the input data is 4D, only the first volume will be displayed.
	- The function assumes that the contour data has non-zero values at specific coordinates.

	"""
	data = np.array(data)
	if data.ndim == 4:
		print("4D volume detected. Only the first volume will be displayed.")
		data = data[:, :, :, 0]

	size_x, size_y, size_z = data.shape
	x, y, z = np.where(data != 55378008)
	coord = np.column_stack((x, y))
	coord = np.column_stack((coord, z))
	coord_array = nib.affines.apply_affine(affine, coord)
	contour_list = np.arange(lthresh, hthresh, ((hthresh - lthresh) / contours)).tolist()
	xi = coord_array[:, 0].reshape(size_x, size_y, size_z)
	yi = coord_array[:, 1].reshape(size_x, size_y, size_z)
	zi = coord_array[:, 2].reshape(size_x, size_y, size_z)
	src = mlab.contour3d(xi, yi, zi, data,
						 vmin=lthresh,
						 vmax=hthresh,
						 opacity=opacity,
						 name=name,
						 contours=contour_list)
	return(src)

# returns the non-empty range
def nonempty_coordinate_range(data, affine):
	nonempty = np.argwhere(data!=0)
	nonempty_native = nib.affines.apply_affine(affine, nonempty)
	x_minmax = np.array((nonempty_native[:,0].min(), nonempty_native[:,0].max()))
	y_minmax = np.array((nonempty_native[:,1].min(), nonempty_native[:,1].max()))
	z_minmax = np.array((nonempty_native[:,2].min(), nonempty_native[:,2].max()))
	return (x_minmax,y_minmax,z_minmax)


# linear function look-up tables
def linear_cm(c0,c1,c2 = None):
	c_map = np.zeros((256,3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128,i] = np.linspace(c0[i],c1[i],128)
			c_map[127:256,i] = np.linspace(c1[i],c2[i],129)
	else:
		for i in range(3):
			c_map[:,i] = np.linspace(c0[i],c1[i],256)
	return(c_map)


# log function look-up tables
def log_cm(c0,c1,c2 = None):
	c_map = np.zeros((256,3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128,i] = np.geomspace(c0[i] + 1,c1[i] + 1,128)-1
			c_map[127:256,i] = np.geomspace(c1[i] + 1,c2[i] + 1,129)-1
	else:
		for i in range(3):
			c_map[:,i] = np.geomspace(c0[i] + 1,c1[i] + 1,256)-1
	return(c_map)


# error function look-up tables
def erf_cm(c0,c1,c2 = None):
	c_map = np.zeros((256,3))
	if c2 is not None:
		for i in range(3):
			c_map[0:128,i] = erf(np.linspace(3*(c0[i]/255),3*(c1[i]/255),128)) * 255
			c_map[127:256,i] = erf(np.linspace(3*(c1[i]/255),3*(c2[i]/255),129)) * 255
	else:
		for i in range(3):
			#c_map[:,i] = erf(np.linspace(0,3,256)) * np.linspace(c0[i], c1[i], 256)
			c_map[:,i] = erf(np.linspace(3*(c0[i]/255),3*(c1[i]/255),256)) * 255 
	return(c_map)

# display the luts included in matplotlib and the customs luts from tmi_viewer
def display_matplotlib_luts():
	# Adapted from https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html

	# This example comes from the Cookbook on www.scipy.org. According to the
	# history, Andrew Straw did the conversion from an old page, but it is
	# unclear who the original author is.

#	plt.switch_backend('Qt4Agg')
	warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

	a = np.linspace(0, 1, 256).reshape(1,-1)
	a = np.vstack((a,a))

	maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
	maps.append('red-yellow') # custom maps 
	maps.append('blue-lightblue')
	maps.append('green-lightgreen')
	maps.append('tm-breeze')
	maps.append('tm-sunset')
	maps.append('tm-broccoli')
	maps.append('tm-octopus')
	maps.append('tm-storm')
	maps.append('tm-flow')
	maps.append('tm-logBluGry')
	maps.append('tm-logRedYel')
	maps.append('tm-erfRGB')
	maps.append('rywlbb')
	maps.append('ryw')
	maps.append('lbb')
	
	nmaps = len(maps) + 1

	fig = plt.figure(figsize=(8,12))
	fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
	for i,m in enumerate(maps):
		ax = plt.subplot(nmaps, 1, i+1)
		plt.axis("off")
		if m == 'red-yellow':
			cmap_array = linear_cm([255,0,0],[255,255,0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'blue-lightblue':
			cmap_array = linear_cm([0,0,255],[0,255,255]) / 255
#			cmap_array = np.array(( np.zeros(256), np.linspace(0,255,256), (np.ones(256)*255) )).T / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'green-lightgreen':
			cmap_array = linear_cm([0,128,0],[0,255,0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-breeze':
			cmap_array = linear_cm([199,233,180],[65,182,196],[37,52,148]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-sunset':
			cmap_array = linear_cm([255,255,51],[255,128,0],[204,0,0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-storm':
			cmap_array = linear_cm([0,153,0],[255,255,0],[204,0,0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-flow':
			cmap_array = log_cm([51,51,255],[255,0,0],[255,255,255]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-logBluGry':
			cmap_array = log_cm([0,0,51],[0,0,255],[255,255,255]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-logRedYel':
			cmap_array = log_cm([102,0,0],[200,0,0],[255,255,0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-erfRGB':
			cmap_array = erf_cm([255,0,0],[0,255,0], [0,0,255]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-broccoli':
			cmap_array = linear_cm([204,255,153],[76,153,0], [0,102,0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'tm-octopus':
			cmap_array = linear_cm([255,204,204],[255,0,255],[102,0,0]) / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'rywlbb':
			cmap_array = create_rywlbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'ryw':
			cmap_array = create_ryw_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'lbb':
			cmap_array = create_lbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		else:
			plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
		pos = list(ax.get_position().bounds)
		fig.text(pos[0] - 0.01, pos[1], m, fontsize=10, horizontalalignment='right')
	plt.show()


# Get RGBA colormap [uint8, uint8, uint8, uint8]
def get_cmap_array(lut, background_alpha = 255, image_alpha = 1.0, zero_lower = True, zero_upper = False, base_color = [227,218,201,0], c_reverse = False):
	base_color[3] = background_alpha
	if lut.endswith('_r'):
		c_reverse = lut.endswith('_r')
		lut = lut[:-2]
	# make custom look-up table
	if (str(lut) == 'r-y') or (str(lut) == 'red-yellow'):
		cmap_array = np.column_stack((linear_cm([255,0,0],[255,255,0]), (255 * np.ones(256) * image_alpha)))
	elif (str(lut) == 'b-lb') or (str(lut) == 'blue-lightblue'):
		cmap_array = np.column_stack((linear_cm([0,0,255],[0,255,255]), (255 * np.ones(256) * image_alpha)))
	elif (str(lut) == 'g-lg') or (str(lut) == 'green-lightgreen'):
		cmap_array = np.column_stack((linear_cm([0,128,0],[0,255,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-breeze':
		cmap_array = np.column_stack((linear_cm([199,233,180],[65,182,196],[37,52,148]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-sunset':
		cmap_array = np.column_stack((linear_cm([255,255,51],[255,128,0],[204,0,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-broccoli':
		cmap_array = np.column_stack((linear_cm([204,255,153],[76,153,0],[0,102,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-octopus':
		cmap_array = np.column_stack((linear_cm([255,204,204],[255,0,255],[102,0,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-storm':
		cmap_array = np.column_stack((linear_cm([0,153,0],[255,255,0],[204,0,0]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-flow':
		cmap_array = np.column_stack((log_cm([51,51,255],[255,0,0],[255,255,255]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-logBluGry':
		cmap_array = np.column_stack((log_cm([0,0,51],[0,0,255],[255,255,255]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-logRedYel':
		cmap_array = np.column_stack((log_cm([102,0,0],[200,0,0],[255,255,0]),(255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-erfRGB':
		cmap_array = np.column_stack((erf_cm([255,0,0],[0,255,0], [0,0,255]), (255 * np.ones(256) * image_alpha)))
	elif str(lut) == 'tm-white':
		cmap_array = np.column_stack((linear_cm([255,255,255],[255,255,255]), (255 * np.ones(256) * image_alpha)))
	else:
		try:
			cmap_array = eval('plt.cm.%s(np.arange(256))' % lut)
			cmap_array[:,3] = cmap_array[:,3] = image_alpha
		except:
			print("Error: Lookup table '%s' is not recognized." % lut)
			print("The lookup table can be red-yellow (r_y), blue-lightblue (b_lb) or any matplotlib colorschemes (https://matplotlib.org/examples/color/colormaps_reference.html)")
			sys.exit()
		cmap_array *= 255
	if c_reverse:
		cmap_array = cmap_array[::-1]
	if zero_lower:
		cmap_array[0] = base_color
	if zero_upper:
		cmap_array[-1] = base_color
	return cmap_array


# Remove black from png
def correct_image(img_name, rotate = None, b_transparent = True, flip = False, crop_black = False):
	img = imageio.imread(img_name)
	if crop_black:
		ind0 = img[:,:,:3].mean(1).mean(1) != 0
		ind1 = img[:,:,:3].mean(0).mean(1) != 0
		img = img[np.min(np.argwhere(ind0)):np.max(np.argwhere(ind0)), np.min(np.argwhere(ind1)):np.max(np.argwhere(ind1)), :]
	if b_transparent:
		if img_name.endswith('.png'):
			rows = img.shape[0]
			columns = img.shape[1]
			if img.shape[2] == 3:
				img_flat = img.reshape([rows * columns, 3])
			else:
				img_flat = img.reshape([rows * columns, 4])
				img_flat = img_flat[:,:3]
			alpha = np.zeros([rows*columns, 1], dtype=np.uint8)
			alpha[img_flat[:,0]!=0] = 255
			alpha[img_flat[:,1]!=0] = 255
			alpha[img_flat[:,2]!=0] = 255
			img_flat = np.column_stack([img_flat, alpha])
			img = img_flat.reshape([rows, columns, 4])
	if rotate is not None:
		img = ndimage.rotate(img, float(rotate))
	if flip:
		img = img[:,::-1,:]
	imageio.imsave(img_name, img)

# add coordinates to the image slices
def add_text_to_img(image_file, add_txt, alpha_int = 200, color = [0,0,0]):
	from PIL import Image, ImageDraw, ImageFont
	base = Image.open(image_file).convert('RGBA')
	txt = Image.new('RGBA', base.size, (255,255,255,0))
	numpixels = base.size[0]
	fnt_size = int(numpixels / 16) # scale the font
	fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', fnt_size)
	start = int(numpixels - (.875 * numpixels))
	d = ImageDraw.Draw(txt)
	d.text((start,start), str(add_txt), font=fnt, fill=(color[0],color[1],color[2],alpha_int))
	out = Image.alpha_composite(base, txt)
	mpimg.imsave(image_file, np.array(out))

# crop and concatenate images
def concate_images(basename, num, clean=False, numpixels = 400):
	start = int(numpixels - (.875 * numpixels))
	stop = int(.875 * numpixels)
	for i in range(num):
		if i == 0:
			outpng = mpimg.imread("%s_0.png" % basename)[start:stop,start:stop,:]
		else:
			tempng = mpimg.imread("%s_%d.png" % (basename, i))[start:stop,start:stop,:]
			outpng = np.concatenate((outpng,tempng),1)
		if i == (num-1):
			mpimg.imsave('%ss.png' % basename, outpng)
	if clean:
		for i in range(num):
			os.remove("%s_%d.png" % (basename, i))

# mask image and draw the outline
def draw_outline(img_png, mask_png, outline_color = [1,0,0,1]):
	from scipy.ndimage.morphology import binary_erosion
	img = mpimg.imread(img_png)
	mask = mpimg.imread(mask_png)
	#check mask
	mask[mask[:,:,3] != 1] = [0,0,0,0]
	mask[mask[:,:,3] == 1] = [1,1,1,1]
	mask[mask[:,:,0] == 1] = [1,1,1,1]
	index = (mask[:,:,0] == 1)
	ones_arr = index *1
	m = ones_arr - binary_erosion(ones_arr)
	index = (m[:,:] == 1)
	img[index] = outline_color
	os.remove(mask_png)
	mpimg.imsave(img_png, img)

# various methods for choosing thresholds automatically
def perform_autothreshold(data, threshold_type = 'yen', z = 2.3264):
	if threshold_type.endswith('_p'):
		data = data[data>0]
	if threshold_type.endswith('_abs'):
		data = data * np.sign(np.mean(data[data!=0]))
		data = data[data>0] 
	else:
		data = data[data!=0]
	if data.size == 0:
		print("Warning: the data array is empty. Auto-thesholding will not be performed")
		return 0, 0
	else:
		if (threshold_type == 'otsu') or (threshold_type == 'otsu_p'):
			lthres = filters.threshold_otsu(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
			# Otsu N (1979) A threshold selection method from gray-level histograms. IEEE Trans. Sys., Man., Cyber. 9: 62-66.
		elif (threshold_type == 'li')  or (threshold_type == 'li_p'):
			lthres = filters.threshold_li(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
			# Li C.H. and Lee C.K. (1993) Minimum Cross Entropy Thresholding Pattern Recognition, 26(4): 617-625
		elif (threshold_type == 'yen') or (threshold_type == 'yen_p'):
			lthres = filters.threshold_yen(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
			# Yen J.C., Chang F.J., and Chang S. (1995) A New Criterion for Automatic Multilevel Thresholding IEEE Trans. on Image Processing, 4(3): 370-378.
		elif threshold_type == 'zscore_p':
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
			if lthres < 0:
				lthres = 0.001
		else:
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
		if uthres > data.max(): # for the rare case when uthres is larger than the max value
			uthres = data.max()
		return(lthres, uthres)

# saves the output dictionary of argparse to a file
def write_dict(filename, outnamespace):
	with open(filename, "wb") as o:
		for k in outnamespace.__dict__:
			if outnamespace.__dict__[k] is not None:
				o.write("%s : %s\n" % (k, outnamespace.__dict__[k]))
		o.close()
