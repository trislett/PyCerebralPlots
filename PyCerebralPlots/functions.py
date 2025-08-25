#!/usr/bin/env python

#	Various functions for PyCerebralPlots
#	Copyright (C) 2023  Tristram Lett

#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.

#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.

#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import sys
import imageio.v2 as imageio
import numpy as np
import warnings
import matplotlib.cbook
import pyvista as pv
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.colorbar import ColorbarBase
from scipy import ndimage
from scipy.special import erf
from scipy.ndimage import convolve
from skimage import filters, measure
from skimage.measure import marching_cubes
from warnings import warn

# get static resources
scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
files_directory = os.path.join(scriptwd, "PyCerebralPlots", "static")
files = os.listdir(files_directory)

annotation_files = [f for f in files if f.endswith(".annot")]
annotation_files = np.unique([f[3:-6] for f in annotation_files])
surfaces_files = [f for f in files if f.endswith(".srf")]
surfaces_files = np.unique([f[3:-4] for f in surfaces_files])
template_files = [f for f in files if f.endswith(".nii.gz")]
template_files = np.unique([f[:-7] for f in template_files])

pack_directory = os.path.join(scriptwd, "PyCerebralPlots", "static", "aseg-subcortical-Surf")
aseg_subcortical_files = np.sort(os.listdir(pack_directory))
pack_directory = os.path.join(scriptwd, "PyCerebralPlots", "static", "JHU-ICBM-Surf")
jhu_white_matter_files = np.sort(os.listdir(pack_directory))

def print_available_neuroimaging_resources():
	print("Available FreeSurfer Surfaces {lh,rh}:")
	for surf in surfaces_files:
		print("\t%s" % surf)
	print("Available FreeSurfer Annotations {lh,rh}:")
	for surf in annotation_files:
		print("\t%s" % surf)
	print("Available Nifti Templates:")
	for surf in template_files:
		print("\t%s" % surf)

def print_available_surface_packs():
	print("Available Freesurfer aseg subcortical pack ['aseg']:")
	for surf in aseg_subcortical_files:
		print("\t%s" % os.path.basename(surf)[:-4])
	print("Available JHU (Mori) white matter tracts ['mori']:")
	for surf in jhu_white_matter_files:
		print("\t%s" % os.path.basename(surf)[:-4])

def get_surface_pack(pack = None):
	if pack is None:
		print("Available packs: {aseg, mori}")
		print_available_surface_packs()
	elif pack == 'aseg':
		return([os.path.join(scriptwd, "PyCerebralPlots", "static", "aseg-subcortical-Surf", surf) for surf in aseg_subcortical_files])
	elif pack == 'mori':
		return([os.path.join(scriptwd, "PyCerebralPlots", "static", "JHU-ICBM-Surf", surf) for surf in jhu_white_matter_files])
	else:
		print("%s not in {aseg, mori}")

def get_neuroimaging_resources(surface_name=None, annotation_name=None, hemisphere=None, template_name=None):
	"""
	Retrieves the file location of neuroimaging resources based on the provided options.

	Parameters
	----------
		surface_name : str:
			Name of the surface file.
		annotation_name : str
			Name of the annotation file.
		hemisphere : str
			Hemisphere ('lh' for left or 'rh' for right).
		template_name : str
			Name of the template file.
	Returns
	-------
		file_loc : str 
			File location of the requested neuroimaging resource.
	Raises
	-------
		ValueError: If more than one option is provided.
		ValueError: If the hemisphere is None or not 'lh' or 'rh'.
		FileNotFoundError: If the file location does not exist.

	"""
	valid_options = ["surface_name", "annotation_name", "template_name"]
	options = [surface_name, annotation_name, template_name]
	counts = sum(option is not None for option in options)

	if counts == 0:
		print_available_neuroimaging_resources()
		return
	if counts > 1:
		raise ValueError(f"Error: include only one of: {', '.join(valid_options)}.")
	if surface_name is not None:
		if hemisphere is None:
			raise ValueError("Error: for surfaces and annotations, the hemisphere cannot be None.")
		if hemisphere not in ['lh', 'rh']:
			raise ValueError("Error: hemisphere must be 'lh' or 'rh'.")
		file_loc = f"{scriptwd}/PyCerebralPlots/static/{hemisphere}.{surface_name}.srf"
	elif annotation_name is not None:
		if hemisphere is None:
			raise ValueError("Error: for surfaces and annotations, the hemisphere cannot be None.")
		if hemisphere not in ['lh', 'rh']:
			raise ValueError("Error: hemisphere must be 'lh' or 'rh'.")
		file_loc = f"{scriptwd}/PyCerebralPlots/static/{hemisphere}.{annotation_name}.annot"
	else:
		file_loc = f"{scriptwd}/PyCerebralPlots/static/{template_name}.nii.gz"
	if os.path.exists(file_loc):
		return(file_loc)
	else:
		raise FileNotFoundError(f"Error: file {file_loc} not found. Select the name from the available resources:")
		print_available_neuroimaging_resources()

# Color maps

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

def create_rywlbb_gradient_cmap(linear_alpha = False, return_array = True):
	colors = ["#00008C", "#2234A8", "#4467C4", "#659BDF", "#87CEFB", "white", "#ffec19", "#ffc100", "#ff9800", "#ff5607", "#f6412d"]
	cmap = LinearSegmentedColormap.from_list("rywlbb-gradient", colors)
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
	cmap = LinearSegmentedColormap.from_list("ryw-gradient", colors)
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
	cmap = LinearSegmentedColormap.from_list("lbb-gradient", colors)
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


# display the luts included in matplotlib and the customs luts from tmi_viewer
def display_matplotlib_luts():
	# Adapted from https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html

	# This example comes from the Cookbook on www.scipy.org. According to the
	# history, Andrew Straw did the conversion from an old page, but it is
	# unclear who the original author is.

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
	maps.append('rywlbb-gradient')
	maps.append('ryw-gradient')
	maps.append('lbb-gradient')
	
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
		elif m == 'rywlbb-gradient':
			cmap_array = create_rywlbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'ryw-gradient':
			cmap_array = create_ryw_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'lbb-gradient':
			cmap_array = create_lbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		else:
			plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
		pos = list(ax.get_position().bounds)
		fig.text(pos[0] - 0.01, pos[1], m, fontsize=10, horizontalalignment='right')
	plt.show()


# Get RGBA colormap [uint8, uint8, uint8, uint8]
def get_cmap_array(lut, background_alpha = 255, image_alpha = 1.0, zero_lower = True, zero_upper = False, base_color = [227,218,201,0], c_reverse = False):
	"""
	Generate an RGBA colormap array based on the specified lookup table (lut) and parameters.
	Use display_matplotlib_luts() to see the available luts.

	Parameters
	----------
	lut : str
		Lookup table name or abbreviation. Accepted values include:
		- 'r-y' or 'red-yellow'
		- 'b-lb' or 'blue-lightblue'
		- 'g-lg' or 'green-lightgreen'
		- 'tm-breeze'
		- 'tm-sunset'
		- 'tm-broccoli'
		- 'tm-octopus'
		- 'tm-storm'
		- 'tm-flow'
		- 'tm-logBluGry'
		- 'tm-logRedYel'
		- 'tm-erfRGB'
		- 'tm-white'
		- 'rywlbb'
		- 'ryw'
		- 'lbb'
		- Any matplotlib colorscheme from https://matplotlib.org/examples/color/colormaps_reference.html
	background_alpha : int, optional
		Alpha value for the background color. Default is 255.
	image_alpha : float, optional
		Alpha value for the colormap colors. Default is 1.0.
	zero_lower : bool, optional
		Whether to set the lower boundary color to the base_color. Default is True.
	zero_upper : bool, optional
		Whether to set the upper boundary color to the base_color. Default is False.
	base_color : list of int, optional
		RGBA values for the base color. Default is [227, 218, 201, 0].
	c_reverse : bool, optional
		Whether to reverse the colormap array. Default is False.

	Returns
	-------
	cmap_array : ndarray
		Custom RGBA colormap array of shape (256, 4) with values in the range of [0, 255].
	"""
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
	elif str(lut) == 'rywlbb-gradient':
		cmap_array = create_rywlbb_gradient_cmap()
	elif str(lut) == 'ryw-gradient':
		cmap_array = create_ryw_gradient_cmap()
	elif str(lut) == 'lbb-gradient':
		cmap_array = create_lbb_gradient_cmap()
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
	return(cmap_array)


def visualize_surface_pack(surf_pack, alpha=1.0, atlas_values=None, vmin=None, vmax=None, 
						  cmap_array=None, alpha_array=None, uniform_lighting=True, 
						  niter_surface_smooth=None, save_figure=None, 
						  save_figure_orientation='xz', output_format='png', 
						  output_transparent_background=True, save_colour_bar=True, 
						  off_screen_render=False, use_lapacian=False):
	"""
	Visualize a collection of brain surfaces with customizable rendering options.
	
	Parameters
	----------
	surf_pack : list of str
		List of file paths to surface files (e.g., FreeSurfer format).
	alpha : float, default=1.0
		Global opacity value for all surfaces (0.0 to 1.0).
	atlas_values : array-like, optional
		Values to color-code each surface. If None, uses linear spacing from 0 to 1.
	vmin, vmax : float, optional
		Minimum and maximum values for color scaling. If None, uses data range.
	cmap_array : array-like, optional
		Custom colormap array (N x 4 RGBA values). If None, uses solid white.
	alpha_array : array-like, optional
		Per-surface opacity values. Must match length of surf_pack.
	uniform_lighting : bool, default=True
		If True, disables lighting for uniform appearance.
	niter_surface_smooth : int, optional
		Number of smoothing iterations to apply to surfaces.
	save_figure : str, optional
		Base filename for saving images. If None, displays interactively.
	save_figure_orientation : str, default='xz'
		Orientations to save ('x', 'y', 'z', 'iso' combinations).
	output_format : str, default='png'
		Output image format ('png', 'jpg', etc.).
	output_transparent_background : bool, default=True
		Whether to use transparent background in saved images.
	save_colour_bar : bool, default=True
		Whether to save a separate colorbar image.
	off_screen_render : bool, default=False
		Whether to render off-screen (headless).
	use_lapacian : bool, default=False
		Whether to use Laplacian smoothing (vs Taubin smoothing).
		
	Returns
	-------
	pyvista.Plotter or None
		Returns plotter object if save_figure is None, otherwise None.
		
	Examples
	--------
	>>> surf_files = get_surface_pack('aseg')
	>>> plotter = visualize_surface_pack(surf_files[:5], alpha=0.8)
	>>> 
	>>> # Save multiple orientations
	>>> visualize_surface_pack(surf_files, save_figure='brain_surfaces',
	...					   save_figure_orientation='xyz')
	"""
	smooth_mode = 'taubin'
	if use_lapacian:
		smooth_mode = 'laplacian'
		
	# Create plotter
	plotter = pv.Plotter(off_screen=off_screen_render or (save_figure is not None))
	
	# Handle atlas values for coloring
	if atlas_values is None:
		atlas_values = np.linspace(0, 1, len(surf_pack))
	else:
		if vmin is None:
			vmin = atlas_values.min()
		if vmax is None:
			vmax = atlas_values.max()
		if (atlas_values.max() > 1.) or (atlas_values.min() < -1.0):
			print("Warning: normalizing the data to min (%1.3f) and max values (%1.3f). It is probably better to set vmin, vmax." % (vmin, vmax))
		atlas_values = (atlas_values - vmin) / (vmax - vmin)
	
	# Set up colormap
	if cmap_array is None:
		cmap_array = np.ones((256, 4), int) * 255
	cmap = ListedColormap((cmap_array / 255.0))
	
	# Handle alpha array
	if alpha_array is None:
		alpha_array = np.ones((len(surf_pack))) * alpha
	else:
		assert len(alpha_array) == len(surf_pack), "Error: the lengths of alpha_array and surf_pack must be the same."
	
	assert np.mean(alpha_array) != 0., "Error: the alpha_array contains only zero values"
	
	# Add surfaces to plotter
	for s, surface_path in enumerate(surf_pack):
		if alpha_array[s] != 0:
			v, f = convert_fs(surface_path)
			if niter_surface_smooth is not None:
				v, f = vectorized_surface_smooth(v, f, adjacency=None, 
											   number_of_iter=niter_surface_smooth, 
											   mode=smooth_mode)
			
			# Create PyVista mesh
			faces = np.column_stack([np.full(f.shape[0], 3), f])
			mesh = pv.PolyData(v, faces)
			
			# Add mesh to plotter
			plotter.add_mesh(mesh, 
						   color=cmap(atlas_values[s])[:3],
						   opacity=alpha_array[s],
						   lighting=not uniform_lighting,
						   smooth_shading=True)

	plotter.set_background('black')
	plotter.camera.parallel_projection = True
	plotter.view_yz()  # Start with left view (equivalent to x_minus_view)

	if save_figure is not None:
		# Axial views (superior and inferior)
		plotter.view_xy(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_axial_superior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		plotter.view_xy(negative=True, render=False)
		plotter.reset_camera()
		savename = f'{save_figure}_axial_inferior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Coronal views (posterior and anterior)
		plotter.view_xz(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_coronal_posterior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		plotter.view_xz(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_coronal_anterior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Sagittal views (right and left)
		plotter.view_yz(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_sagittal_right.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		plotter.view_yz(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_sagittal_left.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		if save_colour_bar:
			write_colorbar(output_basename=save_figure,
						  cmap_array=cmap_array,
						  vmin=vmin,
						  vmax=vmax,
						  colorbar_label=None,
						  output_format='png',
						  abs_colorbar=False,
						  n_ticks=11,
						  orientation='vertical')
		
		plotter.close()
		return None
	else:
		return plotter

def visualize_freesurfer_annotation(surface_path, freesurfer_annotation_path, 
								   atlas_values=None, cmap_array=None, 
								   add_wireframe=True, uniform_lighting=True, 
								   vmin=None, vmax=None, autothreshold_scalar=False, 
								   autothreshold_alg='yen_abs', absmin=None, 
								   absminmax=False, alpha=1.0, niter_surface_smooth=0, 
								   save_figure=None, output_format='png', 
								   output_transparent_background=True, 
								   flat_surface=False, filter_legend_strings=None, 
								   off_screen_render=False, use_lapacian=False):
	"""
	Visualize a FreeSurfer brain surface with annotation-based coloring.
	
	Parameters
	----------
	surface_path : str
		Path to the FreeSurfer surface file.
	freesurfer_annotation_path : str
		Path to the FreeSurfer annotation file.
	atlas_values : array-like, optional
		Custom values for each ROI. If None, uses annotation colors.
	cmap_array : array-like, optional
		Custom colormap array (N x 4 RGBA values). Required if atlas_values provided.
	add_wireframe : bool, default=True
		Whether to highlight annotation boundaries in white.
	uniform_lighting : bool, default=True
		If True, disables lighting for uniform appearance.
	vmin, vmax : float, optional
		Minimum and maximum values for color scaling.
	autothreshold_scalar : bool, default=False
		Whether to automatically threshold scalar values.
	autothreshold_alg : str, default='yen_abs'
		Algorithm for automatic thresholding.
	absmin : float, optional
		Absolute minimum threshold value.
	absminmax : bool, default=False
		Whether to use symmetric min/max values.
	alpha : float, default=1.0
		Opacity value (0.0 to 1.0).
	niter_surface_smooth : int, default=0
		Number of smoothing iterations to apply.
	save_figure : str, optional
		Base filename for saving images. If None, displays interactively.
		Saves all 6 standard orientations (axial, coronal, sagittal views).
	output_format : str, default='png'
		Output image format.
	output_transparent_background : bool, default=True
		Whether to use transparent background in saved images.
	flat_surface : bool, default=False
		Whether to display surface from superior view only.
	filter_legend_strings : str or list, optional
		Strings to filter from annotation names in legend.
	off_screen_render : bool, default=False
		Whether to render off-screen (headless).
	use_lapacian : bool, default=False
		Whether to use Laplacian smoothing (vs Taubin smoothing).
		
	Returns
	-------
	pyvista.Plotter or None
		Returns plotter object if save_figure is None, otherwise None.
		
	Examples
	--------
	>>> surface_file = "path/to/lh.pial"
	>>> annot_file = "path/to/lh.aparc.annot"
	>>> plotter = visualize_freesurfer_annotation(surface_file, annot_file)
	>>> 
	>>> # Save without wireframe boundaries
	>>> visualize_freesurfer_annotation(surface_file, annot_file,
	...							   add_wireframe=False,
	...							   save_figure='brain_annotation')
	"""
	smooth_mode = 'taubin'
	if use_lapacian:
		smooth_mode = 'laplacian'
	
	# Create plotter
	plotter = pv.Plotter(off_screen=off_screen_render or (save_figure is not None))
	
	# Read FreeSurfer annotation
	labels, ctab, names = nib.freesurfer.read_annot(freesurfer_annotation_path)
	roi_indices = np.unique(labels)  # Include background (0) for proper indexing
	
	# Load and optionally smooth surface
	v, f = convert_fs(surface_path)
	if niter_surface_smooth > 0:
		v, f = vectorized_surface_smooth(v, f, adjacency=None, 
									   number_of_iter=niter_surface_smooth, 
									   mode=smooth_mode)
	
	# Create PyVista mesh
	faces = np.column_stack([np.full(f.shape[0], 3), f])
	mesh = pv.PolyData(v, faces)
	
	# Handle coloring based on atlas_values or annotation colors
	if atlas_values is None:
		# Use annotation colors directly
		if -1 in roi_indices:
			roi_indices = roi_indices[1:]
			labels[labels == -1] = 0
		colors = ctab[roi_indices][:, :3]  # Get colors for all unique labels
		
		# Create per-vertex colors
		vertex_colors = np.zeros((len(labels), 3))
		for r, roi in enumerate(roi_indices):
			vertex_colors[labels == roi] = colors[r]
		
		# Add wireframe (boundary highlighting) if requested
		if add_wireframe:
			# Find faces where vertices have different labels (boundaries)
			boundary_faces = [len(set(labels[f[k]])) != 1 for k in range(len(f))]
			boundary_vertices = np.unique(f[boundary_faces])
			# Set boundary vertices to white
			vertex_colors[boundary_vertices] = [169, 169, 169]
		
		# Add colors to mesh
		mesh.point_data['colors'] = vertex_colors.astype(np.uint8)
		
		# Add mesh with vertex colors
		plotter.add_mesh(mesh, 
						scalars='colors',
						rgb=True,  # Use RGB colors directly
						opacity=alpha,
						lighting=not uniform_lighting,
						smooth_shading=True,
						show_scalar_bar=False)
		
		# Create cmap_array for colorbar/legend creation
		cmap_array = np.ones((len(colors), 4), dtype=int) * 255
		cmap_array[:, :3] = colors
		
	else:
		# Use custom atlas values
		assert cmap_array is not None, "Error: a cmap_array must be provided for plotting atlas_values"
		
		roi_indices_nonzero = np.unique(labels)[1:]  # Skip background for atlas values
		scalar_data = np.zeros((len(labels)))
		for r, roi in enumerate(roi_indices_nonzero):
			scalar_data[labels == roi] = atlas_values[r]
		
		if autothreshold_scalar:
			vmin, vmax = perform_autothreshold(scalar_data, threshold_type=autothreshold_alg)
		
		if absmin is not None:
			scalar_data[np.abs(scalar_data) < absmin] = 0
			
		if vmin is None:
			vmin = np.nanmin(scalar_data)
		if vmax is None:
			vmax = np.nanmax(scalar_data)
			
		if absminmax:
			vmax = np.max([np.abs(vmin), np.abs(vmax)])
			vmin = -np.max([np.abs(vmin), np.abs(vmax)])
		
		# Convert scalar values to colors using colormap
		cmap_func = ListedColormap((cmap_array / 255.0))
		# Normalize scalar data to [0, 1] range for colormap
		normalized_data = (scalar_data - vmin) / (vmax - vmin) if vmax != vmin else np.zeros_like(scalar_data)
		vertex_colors = (cmap_func(normalized_data)[:, :3] * 255).astype(np.uint8)
		
		# Add wireframe (boundary highlighting) if requested
		if add_wireframe:
			# Find faces where vertices have different labels (boundaries)
			boundary_faces = [len(set(labels[f[k]])) != 1 for k in range(len(f))]
			boundary_vertices = np.unique(f[boundary_faces])
			# Set boundary vertices to white
			vertex_colors[boundary_vertices] = [169, 169, 169]
		
		# Add colors to mesh
		mesh.point_data['colors'] = vertex_colors
		
		# Add mesh with vertex colors
		plotter.add_mesh(mesh,
						scalars='colors',
						rgb=True,  # Use RGB colors directly
						opacity=alpha,
						lighting=not uniform_lighting,
						smooth_shading=True,
						show_scalar_bar=False)
	
	# Set up scene
	plotter.set_background('black')
	plotter.camera.parallel_projection = True
	
	if flat_surface:
		plotter.view_xy()  # Superior view
	else:
		plotter.view_yz(negative=True)  # Left view
	
	if save_figure is not None:
		if flat_surface:
			plotter.view_xy()
			plotter.reset_camera()
			savename = f'{save_figure}_flat.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		else:
			# Axial views (superior and inferior)
			plotter.view_xy(negative=False)
			plotter.reset_camera()
			savename = f'{save_figure}_axial_superior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			
			plotter.view_xy(negative=True)
			plotter.reset_camera()
			savename = f'{save_figure}_axial_inferior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

			# Coronal views (posterior and anterior)
			plotter.view_xz(negative=False)
			plotter.reset_camera()
			savename = f'{save_figure}_coronal_posterior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			
			plotter.view_xz(negative=True)
			plotter.reset_camera()
			savename = f'{save_figure}_coronal_anterior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

			# Sagittal views (right and left)
			plotter.view_yz(negative=False)
			plotter.reset_camera()
			savename = f'{save_figure}_sagittal_right.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			
			plotter.view_yz(negative=True)
			plotter.reset_camera()
			savename = f'{save_figure}_sagittal_left.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		
		# Save colorbar or legend
		if atlas_values is not None:
			write_colorbar(output_basename=save_figure,
						  cmap_array=cmap_array,
						  vmax=vmax,
						  vmin=vmin,
						  colorbar_label=None,
						  output_format='png',
						  abs_colorbar=False,
						  n_ticks=11,
						  orientation='vertical')
		else:
			names = np.array(names)[roi_indices]  # Use roi_indices instead of np.unique(labels)
			names = [name.decode('utf-8') for name in names]
			if filter_legend_strings is not None:
				if isinstance(filter_legend_strings, str):
					filter_legend_strings = [filter_legend_strings]
				for filter_string in filter_legend_strings:
					names = [name.replace(filter_string, "") for name in names]
			create_annot_legend(labels=names, 
							  rgb_values=ctab[roi_indices][:, :3], 
							  output_basename=save_figure)
		
		plotter.close()
		return None
	else:
		return plotter

def visualize_surface_with_scalar_data(surface, mgh_image=None, cmap_array=None, 
									 vmin=None, vmax=None, autothreshold_scalar=False, 
									 autothreshold_alg='yen_abs', absmin=None, 
									 absminmax=False, alpha=1.0, 
									 render_surface_as_wireframe=False,
									 save_figure=None, output_format='png', 
									 output_transparent_background=True, 
									 color_bar_label=None, niter_surface_smooth=0, 
									 render_annotation_wireframe_path=None, 
									 uniform_lighting=False, off_screen_render=False,
									 use_lapacian=False, flat_surface=False):
	"""
	Renders a FreeSurfer surface with optional scalar data using PyVista. 
	The surface will be interactively rendered if save_figure is None.

	Example usage:
	_ = visualize_surface_with_scalar_data(
		surface=os.environ['SUBJECTS_DIR'] + '/fsaverage/surf/lh.pial_semi_inflated',
		mgh_image="lh_statistic.mgh",
		cmap_array=create_rywlbb_gradient_cmap(),
		autothreshold_scalar=True,
		absminmax=True)

	Parameters
	----------
	surface : str
		The path to the FreeSurfer surface.
	mgh_image : str, optional
		The path to a MGH image file. Defaults to None.
	cmap_array : array, optional
		The colormap array with shape (256,4) and int values ranging from 0 to 255.
	vmin : float, optional
		The minimum value of the scalar data. Defaults to None.
	vmax : float, optional
		The maximum value of the scalar data. Defaults to None.
	autothreshold_scalar : bool, optional
		Whether to perform autothresholding on the scalar data. Defaults to False.
	autothreshold_alg : str, optional
		The autothresholding algorithm to use {yen, otsu, li}. Defaults to 'yen_abs'.
	absmin : float, optional
		The absolute minimum threshold for the scalar data. Defaults to None.
	absminmax : bool, optional
		Whether to use the absolute minimum and maximum values for the colormap range.
	alpha : float, optional
		The level of surface transparency. Defaults to 1.0.
	render_surface_as_wireframe : bool, optional
		Whether to render surface as wireframe. Defaults to False.
	save_figure : str, optional
		The basename string for saving figures. If not None, images and colorbar saved.
	output_format : str, optional
		The output file format for saved figures. Defaults to 'png'.
	output_transparent_background : bool, optional
		Whether to use transparent background in saved images. Defaults to True.
	color_bar_label : str, optional
		The label for the color bar. Defaults to None.
	niter_surface_smooth : int, optional
		The number of iterations for surface smoothing. Defaults to 0.
	render_annotation_wireframe_path : str, optional
		Path to FreeSurfer annotation for wireframe overlay. Defaults to None.
	uniform_lighting : bool, optional
		Whether to disable lighting for uniform appearance. Defaults to False.
	off_screen_render : bool, optional
		Whether to render off-screen (headless). Defaults to False.
	use_lapacian : bool, optional
		Whether to use Laplacian smoothing (vs Taubin). Defaults to False.
	flat_surface : bool, optional
		Whether to display surface from superior view only. Defaults to False.

	Returns
	-------
	pyvista.Plotter or None
		Returns plotter object if save_figure is None, otherwise None.
	"""
	smooth_mode = 'taubin'
	if use_lapacian:
		smooth_mode = 'laplacian'

	# Create plotter
	plotter = pv.Plotter(off_screen=off_screen_render or (save_figure is not None))
	
	if cmap_array is None:
		cmap_array = np.ones((256, 4), int) * 255

	# Load and optionally smooth surface
	v, f = convert_fs(surface)
	
	# Handle annotation wireframe preprocessing if specified
	boundary_vertices = None
	if render_annotation_wireframe_path is not None:
		labels, _, _ = nib.freesurfer.read_annot(render_annotation_wireframe_path)
		# Find faces where vertices have different labels (boundaries)
		boundary_faces = [len(set(labels[f[k]])) != 1 for k in range(len(f))]
		boundary_vertices = np.unique(f[boundary_faces])
	
	if mgh_image is None:
		# No scalar data - just render surface
		if niter_surface_smooth > 0:
			v, f = vectorized_surface_smooth(v, f, adjacency=None, 
										   number_of_iter=niter_surface_smooth, 
										   mode=smooth_mode)
		
		# Create PyVista mesh
		faces = np.column_stack([np.full(f.shape[0], 3), f])
		mesh = pv.PolyData(v, faces)
		
		# Handle wireframe annotation for surface without scalar data
		if boundary_vertices is not None:
			# Create vertex colors (default gray for surface)
			vertex_colors = np.full((len(v), 3), 128, dtype=np.uint8)  # Default gray
			# Set boundary vertices to darker gray
			vertex_colors[boundary_vertices] = [169, 169, 169]
			mesh.point_data['colors'] = vertex_colors
			
			actor = plotter.add_mesh(mesh,
									scalars='colors',
									rgb=True,
									opacity=alpha,
									style='wireframe' if render_surface_as_wireframe else 'surface',
									lighting=not uniform_lighting,
									smooth_shading=True,
									show_scalar_bar=False)
		else:
			# Add mesh without scalars
			actor = plotter.add_mesh(mesh,
									opacity=alpha,
									style='wireframe' if render_surface_as_wireframe else 'surface',
									lighting=not uniform_lighting,
									smooth_shading=True,
									show_scalar_bar=False)
		
	else:
		# Load scalar data from MGH file
		img = nib.load(mgh_image)
		invol = np.asanyarray(img.dataobj)
		scalar_data = check_byteorder(np.squeeze(invol))
		
		if niter_surface_smooth > 0:
			v, f, scalar_data = vectorized_surface_smooth(v, f, adjacency=None, 
														number_of_iter=niter_surface_smooth, 
														scalar=scalar_data, mode=smooth_mode)
		
		if autothreshold_scalar:
			vmin, vmax = perform_autothreshold(scalar_data, threshold_type=autothreshold_alg)
			
		if absmin is not None:
			scalar_data[np.abs(scalar_data) < absmin] = 0
			
		if vmin is None:
			vmin = np.nanmin(scalar_data)
		if vmax is None:
			vmax = np.nanmax(scalar_data)
			
		if absminmax:
			vmax = np.max([np.abs(vmin), np.abs(vmax)])
			vmin = -np.max([np.abs(vmin), np.abs(vmax)])

		# Create PyVista mesh
		faces = np.column_stack([np.full(f.shape[0], 3), f])
		mesh = pv.PolyData(v, faces)
		
		# Convert scalar values to colors using colormap
		from matplotlib.colors import ListedColormap
		cmap_func = ListedColormap((cmap_array / 255.0))
		
		# Normalize scalar data to [0, 1] range for colormap
		if vmax != vmin:
			normalized_data = (scalar_data - vmin) / (vmax - vmin)
		else:
			normalized_data = np.zeros_like(scalar_data)
		
		# Apply colormap to get vertex colors
		vertex_colors = (cmap_func(normalized_data)[:, :3] * 255).astype(np.uint8)
		
		# Add wireframe annotation boundaries if specified
		if boundary_vertices is not None:
			vertex_colors[boundary_vertices] = [169, 169, 169]
		
		# Add colors to mesh
		mesh.point_data['colors'] = vertex_colors
		
		# Add mesh with colors
		actor = plotter.add_mesh(mesh,
								scalars='colors',
								rgb=True,
								opacity=alpha,
								style='wireframe' if render_surface_as_wireframe else 'surface',
								lighting=not uniform_lighting,
								smooth_shading=True,
								show_scalar_bar=False)

	# Set up scene
	plotter.set_background('black')
	plotter.camera.parallel_projection = True
	
	if flat_surface:
		plotter.view_xy()  # Superior view
	else:
		plotter.view_yz(negative=True)  # Left view

	if save_figure is not None:
		if flat_surface:
			plotter.view_xy()
			plotter.reset_camera()
			savename = f'{save_figure}_flat.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		else:
			# Axial views (superior and inferior)
			plotter.view_xy(negative=False)
			plotter.reset_camera()
			savename = f'{save_figure}_axial_superior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			
			plotter.view_xy(negative=True)
			plotter.reset_camera()
			savename = f'{save_figure}_axial_inferior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

			# Coronal views (posterior and anterior)
			plotter.view_xz(negative=False)
			plotter.reset_camera()
			savename = f'{save_figure}_coronal_posterior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			
			plotter.view_xz(negative=True)
			plotter.reset_camera()
			savename = f'{save_figure}_coronal_anterior.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

			# Sagittal views (right and left)
			plotter.view_yz(negative=False)
			plotter.reset_camera()
			savename = f'{save_figure}_sagittal_right.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
			
			plotter.view_yz(negative=True)
			plotter.reset_camera()
			savename = f'{save_figure}_sagittal_left.{output_format}'
			plotter.screenshot(savename, transparent_background=output_transparent_background)
			correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Save colorbar if we have scalar data
		if mgh_image is not None:
			write_colorbar(output_basename=save_figure,
						  cmap_array=cmap_array,
						  vmax=vmax,
						  vmin=vmin,
						  colorbar_label=color_bar_label,
						  output_format='png',
						  abs_colorbar=False,
						  n_ticks=11,
						  orientation='vertical')

		plotter.close()
		return None
	else:
		return plotter

def visualize_cube_voxel_with_scalar_data(
						voxel_data_positive, voxel_data_negative=None, threshold=0.95, 
						vmin=0, vmax=1.0, clip=True, voxel_alpha=0.7, 
						positive_cmap='red-yellow', negative_cmap='blue-lightblue', 
						surface_names=None, surface_alpha=0.2, mask_data=None, 
						mask_color='white', mask_alpha=0.2,
						save_figure=None, output_format='png', 
						output_transparent_background=True, color_bar_label=None,
						off_screen_render=False):
	"""
	Convert voxel-wise MRI statistics to a 3D mesh representation with brain surfaces and mask surface.
	Optimized version using batch processing.
	"""
	from skimage.measure import marching_cubes
	from scipy.ndimage import convolve
	
	# Get colormap arrays and create matplotlib colormaps
	cmap_positive = ListedColormap(get_cmap_array(positive_cmap)/255)
	cmap_negative = ListedColormap(get_cmap_array(negative_cmap)/255)

	# Create plotter
	plotter = pv.Plotter(off_screen=off_screen_render or (save_figure is not None))
	
	def create_batch_cubes(indices, values, cmap, cube_size=1.0):
		"""Create multiple cubes efficiently as a single mesh"""
		if len(indices) == 0:
			return None
		
		# Create a single cube template
		cube_template = pv.Cube(x_length=cube_size, y_length=cube_size, z_length=cube_size)
		cube_points = cube_template.points
		
		# Parse PyVista faces format properly
		faces_raw = cube_template.faces
		cube_faces = []
		i = 0
		while i < len(faces_raw):
			n_verts = faces_raw[i]
			face = faces_raw[i+1:i+1+n_verts]
			cube_faces.append(face)
			i += n_verts + 1
		cube_faces = np.array(cube_faces)
		
		n_cubes = len(indices)
		n_points_per_cube = len(cube_points)
		n_faces_per_cube = len(cube_faces)
		
		# Pre-allocate arrays
		all_points = np.zeros((n_cubes * n_points_per_cube, 3))
		all_colors = np.zeros((n_cubes * n_points_per_cube, 3))
		
		# Build faces list
		faces_list = []
		
		# Batch process all cubes
		for i, (idx, val) in enumerate(zip(indices, values)):
			x, y, z = idx
			
			# Translate cube points to position
			start_pt = i * n_points_per_cube
			end_pt = start_pt + n_points_per_cube
			all_points[start_pt:end_pt] = cube_points + [x, y, z]
			
			# Add faces for this cube
			for face in cube_faces:
				face_with_count = [len(face)] + (face + start_pt).tolist()
				faces_list.extend(face_with_count)
			
			# Set colors for all points of this cube
			color = cmap(val)[:3]
			all_colors[start_pt:end_pt] = color
		
		# Create mesh
		mesh = pv.PolyData(all_points, faces_list)
		mesh.point_data['colors'] = (all_colors * 255).astype(np.uint8)
		
		return mesh
	
	# Process positive voxel data
	pos_indices = np.where(voxel_data_positive > threshold)
	if len(pos_indices[0]) > 0:
		pos_coords = list(zip(pos_indices[0], pos_indices[1], pos_indices[2]))
		
		# Get and normalize values
		if clip:
			pos_values = np.clip(voxel_data_positive[pos_indices], threshold, vmax)
		else:
			pos_values = voxel_data_positive[pos_indices]
		
		pos_norm_values = (pos_values - threshold) / (vmax - threshold)
		
		# Create batch mesh
		pos_mesh = create_batch_cubes(pos_coords, pos_norm_values, cmap_positive)
		if pos_mesh is not None:
			plotter.add_mesh(pos_mesh, scalars='colors', rgb=True, 
							opacity=voxel_alpha, show_scalar_bar=False)
	
	# Process negative voxel data if provided
	if voxel_data_negative is not None:
		neg_indices = np.where(voxel_data_negative > threshold)
		if len(neg_indices[0]) > 0:
			neg_coords = list(zip(neg_indices[0], neg_indices[1], neg_indices[2]))
			
			if clip:
				neg_values = np.clip(voxel_data_negative[neg_indices], threshold, vmax)
			else:
				neg_values = voxel_data_negative[neg_indices]
			
			neg_norm_values = (neg_values - threshold) / (vmax - threshold)
			
			# Create batch mesh
			neg_mesh = create_batch_cubes(neg_coords, neg_norm_values, cmap_negative)
			if neg_mesh is not None:
				plotter.add_mesh(neg_mesh, scalars='colors', rgb=True, 
								opacity=voxel_alpha, show_scalar_bar=False)
	
	# Add brain surfaces if provided
	if surface_names:
		for sn in surface_names:
			vertices, faces = load_surface_geometry(sn)
			faces_pv = np.column_stack((np.full(len(faces), 3), faces)).ravel()
			mesh = pv.PolyData(vertices, faces_pv)
			plotter.add_mesh(mesh, color='lightgray', opacity=surface_alpha, 
							smooth_shading=True, show_scalar_bar=False)
	
	# Add mask as a surface if provided
	if mask_data is not None:
		binary_mask = (mask_data > 0).astype(np.int8)
		try:
			verts, faces, normals, values = marching_cubes(binary_mask, level=0.5)
			mask_faces = np.column_stack((np.full(len(faces), 3), faces))
			mask_mesh = pv.PolyData(verts, mask_faces)
			plotter.add_mesh(mask_mesh, color=mask_color, opacity=mask_alpha, 
							smooth_shading=True, show_scalar_bar=False)
			
		except Exception as e:
			print(f"Error creating mask surface: {e}")
			print("Falling back to simple mask outline...")
			kernel = np.ones((3, 3, 3), dtype=np.int8)
			kernel[1, 1, 1] = 0  
			neighbor_count = convolve(binary_mask, kernel, mode='constant', cval=0)
			boundary_mask = (binary_mask > 0) & (neighbor_count < 26)
			bx, by, bz = np.where(boundary_mask)
			points = np.column_stack((bx, by, bz))
			if len(points) > 0:
				boundary_point_cloud = pv.PolyData(points)
				plotter.add_mesh(boundary_point_cloud, color=mask_color, 
								point_size=5, render_points_as_spheres=True,
								show_scalar_bar=False)

	# Set up scene
	plotter.set_background('black')
	plotter.camera.parallel_projection = True
	plotter.view_yz(negative=True)  # Left view

	if save_figure is not None:
		# Axial views (superior and inferior)
		plotter.view_xy(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_axial_superior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		
		plotter.view_xy(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_axial_inferior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Coronal views (posterior and anterior)
		plotter.view_xz(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_coronal_posterior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		
		plotter.view_xz(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_coronal_anterior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Sagittal views (right and left)
		plotter.view_yz(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_sagittal_right.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		
		plotter.view_yz(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_sagittal_left.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Save colorbar - determine which colormap to use for colorbar
		# Use positive colormap if it has data, otherwise negative
		if len(pos_indices[0]) > 0:
			cmap_array = get_cmap_array(positive_cmap)
		elif voxel_data_negative is not None and len(neg_indices[0]) > 0:
			cmap_array = get_cmap_array(negative_cmap)
		else:
			cmap_array = get_cmap_array(positive_cmap)  # fallback
		
		write_colorbar(output_basename=save_figure,
					  cmap_array=cmap_array,
					  vmax=vmax,
					  vmin=threshold,  # Use threshold as vmin for colorbar
					  colorbar_label=color_bar_label,
					  output_format='png',
					  abs_colorbar=False,
					  n_ticks=11,
					  orientation='vertical')

		plotter.close()
		return None
	else:
		return plotter

def visualize_volume_to_surface(nifti_image_path, cmap_array=None, nifti_image_mask=None, 
							   volume_alpha=0.8, vmin=None, vmax=None, 
							   autothreshold_scalar=False, autothreshold_alg='yen_abs', 
							   absmin=None, absminmax=False, render_mask_volume=None, 
							   binarize_render_mask_volume=True, render_mask_volume_alpha=0.8, 
							   save_figure=None, output_format='png', 
							   output_transparent_background=True, color_bar_label=None, 
							   niter_surface_smooth=0, smooth_scalar=False, 
							   niter_surface_smooth_render_mask=0, off_screen_render=False, 
							   save_colour_bar=True):
	"""
	Visualizes a NIfTI volume as a surface. The voxels are converted to a surface using 
	a marching cube algorithm and surface then undergoes an affine transformation.
	If save_figure is None, an interactive visualization is displayed.
	
	Example usage:
		nifti_image_path = '/path/to/example_statistic_nifti.nii.gz'
		render_mask_volume = '/path/to/example_template_brain.nii.gz'
		visualize_volume_to_surface(nifti_image_path=nifti_image_path,
									cmap_array=create_rywlbb_gradient_cmap(),
									absminmax=True,
									render_mask_volume=render_mask_volume,
									save_figure='example_figure',
									autothreshold_scalar=True,
									autothreshold_alg='yen_abs',
									niter_surface_smooth=0,
									niter_surface_smooth_render_mask=8)
	
	Parameters
	----------
	nifti_image_path : str
		Path to the NIfTI image file.
	cmap_array : numpy.ndarray, optional
		Colormap array for the surface visualization.
	nifti_image_mask : str, optional
		Path to the NIfTI mask file. (default: None)
	volume_alpha : float, optional
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
	render_mask_volume_alpha : float, optional
		Opacity of the render mask volume visualization. (default: 0.8)
	save_figure : str, optional
		Path to save the figure. (default: None)
	output_format : str, optional
		Format of the saved figure. (default: 'png')
	output_transparent_background : bool, optional
		Flag to make the saved figure's background transparent. (default: True)
	color_bar_label : str, optional
		Label for the colorbar. (default: None)
	niter_surface_smooth : int, optional
		Number of iterations for smoothing the surface. (default: 0)
	smooth_scalar : bool, optional
		Whether to smooth scalar values along with surface. (default: False)
	niter_surface_smooth_render_mask : int, optional
		Number of iterations for smoothing the render mask volume. (default: 0)
	off_screen_render : bool, optional
		Whether to render off-screen (headless). (default: False)
	save_colour_bar : bool, optional
		Whether to save the colorbar. (default: True)

	Returns
	-------
	pyvista.Plotter or None
		Returns plotter object if save_figure is None, otherwise None.
	"""
	# Create plotter
	plotter = pv.Plotter(off_screen=off_screen_render or (save_figure is not None))
	
	if cmap_array is None:
		cmap_array = np.ones((256, 4), int) * 255

	# Load NIfTI data
	invol = nib.as_closest_canonical(nib.load(nifti_image_path))
	data = check_byteorder(np.asanyarray(invol.dataobj))
	
	# Handle masking
	if nifti_image_mask is None:
		mask_arr = data != 0
	else:
		mask = nib.as_closest_canonical(nib.load(nifti_image_mask))
		mask_arr = check_byteorder(np.asanyarray(mask.dataobj)) != 0
	
	scalar_data = data[mask_arr]
	
	# Handle thresholding and value ranges
	if autothreshold_scalar:
		vmin, vmax = perform_autothreshold(scalar_data, threshold_type=autothreshold_alg)
	if absmin is not None:
		scalar_data[np.abs(scalar_data) < absmin] = 0
	if vmin is None:
		vmin = np.nanmin(scalar_data)
	if vmax is None:
		vmax = np.nanmax(scalar_data)
	if absminmax:
		vmax = np.max([np.abs(vmin), np.abs(vmax)])
		vmin = -np.max([np.abs(vmin), np.abs(vmax)])

	# Convert scalar values to colors using colormap
	from matplotlib.colors import ListedColormap
	cmap_func = ListedColormap((cmap_array / 255.0))

	# Handle positive data
	if np.max(data) > 0:
		data_pos = np.array(data)
		data_pos[data_pos < 0] = 0
		v, f, scalar_data_pos = convert_voxel(data_pos, affine=invol.affine)
		
		if niter_surface_smooth > 0:
			if smooth_scalar:
				v, f, scalar_data_pos = vectorized_surface_smooth(v, f, adjacency=None, 
																number_of_iter=niter_surface_smooth, 
																scalar=scalar_data_pos, mode='taubin')
			else:
				v, f = vectorized_surface_smooth(v, f, adjacency=None, 
											   number_of_iter=niter_surface_smooth, 
											   scalar=None, mode='taubin')
		
		# Create PyVista mesh for positive surface
		faces = np.column_stack([np.full(f.shape[0], 3), f])
		mesh_pos = pv.PolyData(v, faces)
		
		# Normalize scalar data and apply colormap
		if vmax != vmin:
			normalized_data = np.clip((scalar_data_pos - vmin) / (vmax - vmin), 0, 1)
		else:
			normalized_data = np.zeros_like(scalar_data_pos)
		
		vertex_colors = (cmap_func(normalized_data)[:, :3] * 255).astype(np.uint8)
		mesh_pos.point_data['colors'] = vertex_colors
		
		# Add positive surface
		plotter.add_mesh(mesh_pos,
						scalars='colors',
						rgb=True,
						opacity=volume_alpha,
						smooth_shading=True,
						show_scalar_bar=False)

	# Handle negative data
	if np.min(data) < 0:
		data_neg = np.array(data)
		data_neg[data_neg > 0] = 0
		v, f, scalar_data_neg = convert_voxel(data_neg * -1, affine=invol.affine)
		
		if niter_surface_smooth > 0:
			if smooth_scalar:
				v, f, scalar_data_neg = vectorized_surface_smooth(v, f, adjacency=None, 
																number_of_iter=niter_surface_smooth, 
																scalar=scalar_data_neg, mode='taubin')
			else:
				v, f = vectorized_surface_smooth(v, f, adjacency=None, 
											   number_of_iter=niter_surface_smooth, 
											   scalar=None, mode='taubin')
		
		# Create PyVista mesh for negative surface
		faces = np.column_stack([np.full(f.shape[0], 3), f])
		mesh_neg = pv.PolyData(v, faces)
		
		# Negate scalars and normalize
		neg_scalars = -scalar_data_neg
		if vmax != vmin:
			normalized_data = np.clip((neg_scalars - vmin) / (vmax - vmin), 0, 1)
		else:
			normalized_data = np.zeros_like(neg_scalars)
		
		vertex_colors = (cmap_func(normalized_data)[:, :3] * 255).astype(np.uint8)
		mesh_neg.point_data['colors'] = vertex_colors
		
		# Add negative surface
		plotter.add_mesh(mesh_neg,
						scalars='colors',
						rgb=True,
						opacity=volume_alpha,
						smooth_shading=True,
						show_scalar_bar=False)

	# Render mask volume if provided
	if render_mask_volume is not None:
		render_mask = nib.as_closest_canonical(nib.load(render_mask_volume))
		mask_data = check_byteorder(np.asanyarray(render_mask.dataobj))
		
		if binarize_render_mask_volume:
			mask_data[mask_data != 0] = 1
		
		# Convert mask volume to surface
		v, f, scalar_mask_data = convert_voxel(mask_data, affine=render_mask.affine)
		
		# Apply smoothing if requested
		if niter_surface_smooth_render_mask > 0:
			v, f = vectorized_surface_smooth(v, f, adjacency=None, 
										   number_of_iter=niter_surface_smooth_render_mask, 
										   scalar=None, mode='taubin')
		
		# Create PyVista mesh for mask
		faces = np.column_stack([np.full(f.shape[0], 3), f])
		mask_mesh = pv.PolyData(v, faces)
		mask_mesh.point_data['scalars'] = scalar_mask_data
		
		# Add mask mesh with white/gray appearance
		plotter.add_mesh(mask_mesh,
						color='lightgray',
						opacity=render_mask_volume_alpha,
						smooth_shading=True,
						show_scalar_bar=False)

	# Set up scene
	plotter.set_background('black')
	plotter.camera.parallel_projection = True
	plotter.view_yz(negative=True)  # Left view

	if save_figure is not None:
		# Axial views (superior and inferior)
		plotter.view_xy(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_axial_superior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		
		plotter.view_xy(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_axial_inferior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Coronal views (posterior and anterior)
		plotter.view_xz(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_coronal_posterior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		
		plotter.view_xz(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_coronal_anterior.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Sagittal views (right and left)
		plotter.view_yz(negative=False)
		plotter.reset_camera()
		savename = f'{save_figure}_sagittal_right.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)
		
		plotter.view_yz(negative=True)
		plotter.reset_camera()
		savename = f'{save_figure}_sagittal_left.{output_format}'
		plotter.screenshot(savename, transparent_background=output_transparent_background)
		correct_image(savename, crop_black=True, b_transparent=output_transparent_background)

		# Save colorbar if requested
		if save_colour_bar:
			write_colorbar(output_basename=save_figure,
						  cmap_array=cmap_array,
						  vmin=vmin,
						  vmax=vmax,
						  colorbar_label=color_bar_label,
						  output_format='png',
						  abs_colorbar=False,
						  n_ticks=11,
						  orientation='vertical')

		plotter.close()
		return None
	else:
		return plotter


# helper functions
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

def label_to_surface(labels, values, masked = None, null_value = -1, make_3d = False):
	outvalue = np.array(values)
	if masked is not None:
		outvalue[masked] = null_value
	outdata = np.zeros_like(labels, dtype=float)
	for i, value in enumerate(outvalue):
		label_num = i+1
		index_arr = labels == label_num
		outdata[index_arr] = value
	if make_3d:
		outdata = outdata[:, np.newaxis, np.newaxis]
	return outdata.astype(np.float32, order = "C")

def _plot_colormap(cmap):
	data = np.arange(100).reshape((10, 10))  # Example data for the plot
	fig, ax = plt.subplots()
	im = ax.imshow(data, cmap=cmap)
	fig.colorbar(im)
	plt.show()

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

def vectorized_surface_smooth(v, f, adjacency = None, number_of_iter = 5, scalar = None, lambda_w = 0.5, mode = 'taubin', weighted = True):
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
	number_of_iter : int
		number of smoothing iterations
	scalar : array
		apply the same smoothing to a image scalar
	lambda_w : float
		lamda weighting of degree of movement for each iteration
		The weighting should never be above 1.0
	mode : string
		The type of smoothing can either be laplacian (which cause surface shrinkage) or taubin (no shrinkage). Default is 'taubin'
		
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
	
	# Control of damping effects
	k = 0.1
	lambda_w = lambda_w/(1-k*lambda_w)
	mu_w = -lambda_w

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

def crop_image_alpha(img_name):
	"""
	Crops png image based on alpha = 0.
	"""
	img = imageio.imread(img_name)
	ind0 = img[:,:,3].mean(1) != 0
	ind1 = img[:,:,3].mean(0) != 0
	img = img[np.min(np.argwhere(ind0)):np.max(np.argwhere(ind0)), np.min(np.argwhere(ind1)):np.max(np.argwhere(ind1)), :]
	imageio.imsave(img_name, img)

def create_annot_legend(labels, rgb_values, output_basename, num_columns = None, output_format = 'png', ratio = 4):
	num_boxes = len(labels)
	if num_columns is None:
		num_columns = int(np.sqrt(num_boxes / (ratio+1)))
	num_rows = np.ceil(num_boxes / num_columns)
	fig, ax = plt.subplots(figsize=(num_columns * 10, num_rows))
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
	crop_image_alpha("%s_annot_legend.%s" % (output_basename, output_format))
	plt.close()

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

def write_colorbar(output_basename, cmap_array, vmax, vmin=None, colorbar_label=None, 
				  output_format='png', abs_colorbar=False, n_ticks=11, orientation='vertical'):
	"""
	Generate a colorbar and save it as an image file.

	Parameters
	----------
	output_basename : str 
		Base name of the output file.
	cmap_array : ndarray 
		Array of colors representing the colormap.
	vmax : float
		Maximum value for the colorbar.
	vmin : float, optional
		Minimum value for the colorbar. If not provided, it defaults to 0.
	colorbar_label : str, optional
		Label for the colorbar.
	output_format : str, optional
		Output file format. Defaults to 'png'.
	abs_colorbar : bool, optional
		Whether to treat the colorbar as absolute, making vmin = -vmax if True. Defaults to False.
	n_ticks : int, optional
		Number of ticks to be displayed on the colorbar. Defaults to 11.
	orientation : str, optional
		Orientation of the colorbar ('vertical' or 'horizontal'). Defaults to 'vertical'.
		
	Returns
	-------
	None
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.colors import ListedColormap
	from matplotlib.colorbar import ColorbarBase
	
	if abs_colorbar:
		vmin = -vmax
	if vmin is None:
		vmin = 0
		
	cmap = ListedColormap(np.divide(cmap_array, 255))
	
	if orientation == 'horizontal':
		fig, ax = plt.subplots(figsize=(4, 1))
	else:
		fig, ax = plt.subplots(figsize=(1, 4))
		
	colorbar = ColorbarBase(ax, cmap=cmap, orientation=orientation)
	
	if colorbar_label is not None:
		colorbar.set_label(colorbar_label)
		
	tick_labels = ["%1.2f" % t for t in np.linspace(vmin, vmax, n_ticks)]
	colorbar.set_ticks(ticks=np.linspace(0, 1, n_ticks), labels=tick_labels)
	
	plt.tight_layout()
	plt.savefig("%s_colorbar.%s" % (output_basename, output_format), transparent=True)
	plt.close()

def apply_affine_to_scalar_field(data, affine):
	"""
	Creates a PyVista ImageData object with applied affine transformation.
	
	Parameters
	----------
	data : array-like
		The input scalar field data.
	affine : array-like
		The affine transformation matrix to be applied.
		
	Returns
	-------
	volume : pv.ImageData
		The PyVista ImageData object with transformed coordinates.
	"""
	data = np.array(data)
	if data.ndim == 4:
		print("4D volume detected. Only the first volume will be displayed.")
		data = data[:,:,:,0]
	
	# Get spacing and origin from affine
	spacing = np.abs(np.diag(affine)[:3])
	origin = affine[:3, 3]
	
	# Create PyVista ImageData
	volume = pv.ImageData(dimensions=data.shape[::-1], spacing=spacing[::-1], origin=origin[::-1])
	volume.point_data['scalars'] = data.ravel(order='F')
	
	return volume

def apply_affine_to_contour3d(data, affine, lthresh, hthresh, name="", contours=15, opacity=0.7):
	"""
	Creates contour surfaces from 3D data with affine transformation using PyVista.
	
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
	contour_surfaces : pv.PolyData
		The contour surfaces with transformed coordinates.
	"""
	data = np.array(data)
	if data.ndim == 4:
		print("4D volume detected. Only the first volume will be displayed.")
		data = data[:, :, :, 0]
	
	# Get spacing and origin from affine
	spacing = np.abs(np.diag(affine)[:3])
	origin = affine[:3, 3]
	
	# Create PyVista ImageData
	volume = pv.ImageData(dimensions=data.shape[::-1], spacing=spacing[::-1], origin=origin[::-1])
	volume.point_data['scalars'] = data.ravel(order='F')
	
	# Create contour levels
	contour_list = np.linspace(lthresh, hthresh, contours)
	
	# Generate contours
	contour_surfaces = volume.contour(isosurfaces=contour_list, scalars='scalars')
	
	return contour_surfaces

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
	maps.append('rywlbb-gradient')
	maps.append('ryw-gradient')
	maps.append('lbb-gradient')
	
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
		elif m == 'rywlbb-gradient':
			cmap_array = create_rywlbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'ryw-gradient':
			cmap_array = create_ryw_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		elif m == 'lbb-gradient':
			cmap_array = create_lbb_gradient_cmap() / 255
			plt.imshow(a, aspect='auto', cmap=ListedColormap(cmap_array,m), origin='lower')
		else:
			plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
		pos = list(ax.get_position().bounds)
		fig.text(pos[0] - 0.01, pos[1], m, fontsize=10, horizontalalignment='right')
	plt.show()


# Get RGBA colormap [uint8, uint8, uint8, uint8]
def get_cmap_array(lut, background_alpha = 255, image_alpha = 1.0, zero_lower = True, zero_upper = False, base_color = [227,218,201,0], c_reverse = False):
	"""
	Generate an RGBA colormap array based on the specified lookup table (lut) and parameters.
	Use display_matplotlib_luts() to see the available luts.

	Parameters
	----------
	lut : str
		Lookup table name or abbreviation. Accepted values include:
		- 'r-y' or 'red-yellow'
		- 'b-lb' or 'blue-lightblue'
		- 'g-lg' or 'green-lightgreen'
		- 'tm-breeze'
		- 'tm-sunset'
		- 'tm-broccoli'
		- 'tm-octopus'
		- 'tm-storm'
		- 'tm-flow'
		- 'tm-logBluGry'
		- 'tm-logRedYel'
		- 'tm-erfRGB'
		- 'tm-white'
		- 'rywlbb'
		- 'ryw'
		- 'lbb'
		- Any matplotlib colorscheme from https://matplotlib.org/examples/color/colormaps_reference.html
	background_alpha : int, optional
		Alpha value for the background color. Default is 255.
	image_alpha : float, optional
		Alpha value for the colormap colors. Default is 1.0.
	zero_lower : bool, optional
		Whether to set the lower boundary color to the base_color. Default is True.
	zero_upper : bool, optional
		Whether to set the upper boundary color to the base_color. Default is False.
	base_color : list of int, optional
		RGBA values for the base color. Default is [227, 218, 201, 0].
	c_reverse : bool, optional
		Whether to reverse the colormap array. Default is False.

	Returns
	-------
	cmap_array : ndarray
		Custom RGBA colormap array of shape (256, 4) with values in the range of [0, 255].
	"""
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
	elif str(lut) == 'rywlbb-gradient':
		cmap_array = create_rywlbb_gradient_cmap()
	elif str(lut) == 'ryw-gradient':
		cmap_array = create_ryw_gradient_cmap()
	elif str(lut) == 'lbb-gradient':
		cmap_array = create_lbb_gradient_cmap()
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
	return(cmap_array)


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

def load_surface_geometry(path_to_surface):
	"""
	Load surface geometry (vertices and faces) from various neuroimaging file formats.
	
	Parameters
	----------
	path_to_surface : str
		Path to surface file. Supported formats:
		- FreeSurfer (.srf)
		- GIFTI (.surf.gii)
		- CIFTI (.d*.nii)
		- VTK (.vtk)
	
	Returns
	-------
	v : np.ndarray
		Vertex coordinates (N, 3)
	f : np.ndarray
		Face connectivity indices (M, 3)
	
	Raises
	------
	ValueError
		If file format is not supported or surface data cannot be extracted
	"""
	if not os.path.exists(path_to_surface):
		raise FileNotFoundError("Surface file not found: [%s]" % path_to_surface)
	ext = os.path.splitext(path_to_surface)[1].lower()

	# extra checks for ext
	if path_to_surface.endswith('.dtseries.nii') or path_to_surface.endswith('.dtseries.nii.gz'):
		ext = '.dtseries.nii'
	if path_to_surface.endswith('.dscalar.nii') or path_to_surface.endswith('.dscalar.nii.gz'):
		ext = '.dscalar.nii'
	if path_to_surface.endswith('.dlabel.nii') or path_to_surface.endswith('.dlabel.nii.gz'):
		ext = '.dlabel.nii'
	if ext == '.srf':
		v, f = nib.freesurfer.io.read_geometry(path_to_surface)
		return(v, f)
	elif ext == '.gii':
		gii = nib.load(path_to_surface)
		v, f = None, None
		for da in gii.darrays:
			if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
				v = da.data
			elif da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
				f = da.data
		if v is None or f is None:
			raise ValueError("GIFTI file missing vertex or face data")
		return(v, f)
	elif ext in ('.dtseries.nii', '.dscalar.nii', '.dlabel.nii'):
		if path_to_surface.endswith('.gz'):
			outfile = 'tempfile.nii'
			with gzip.open(path_to_surface, 'rb') as file_in:
				with open(outfile, 'wb') as file_out:
					shutil.copyfileobj(file_in, file_out)
			cifti = nib.load(outfile)
			os.remove(outfile)
		else:
			cifti = nib.load(path_to_surface)
		for brain_model in cifti.header.get_axis(1).iter_structures():
			print(brain_model)
			if brain_model[0] == 'CIFTI_MODEL_TYPE_SURFACE':
				v, f = brain_model[1].surface.surface_vertices, brain_model[1].surface.surface_faces
				return(v, f)
		raise ValueError("No surface data found in CIFTI file")
	elif ext == '.vtk':
		mesh = pv.read(path_to_surface)
		v = mesh.points
		f = mesh.faces.reshape(-1, 4)[:, 1:4]
		return(v, f)
	else:
		print("Warning: attempting to load [%s] as freesurfer surface mesh" % path_to_surface)
		try:
			v, f = nib.freesurfer.io.read_geometry(path_to_surface)
			return(v, f)
		except:
			raise ValueError("Unsupported surface file [%s] " % path_to_surface)