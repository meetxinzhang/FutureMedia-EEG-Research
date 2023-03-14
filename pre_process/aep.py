# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/3/2 21:04
 @desc:
  Forked from https://github.com/numediart/EEGLearn-Pytorch
  Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International
  conference on learning representations (2016).
"""

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
import math as m
import numpy as np
np.random.seed(123)
import einops
from sklearn.decomposition import PCA


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)


def augment_EEG(data, stdMult, pca=False, n_components=2):
    """
    Augment data by adding normal noise to each feature.
    :param data: EEG feature data as a matrix (n_samples x n_features)
    :param stdMult: Multiplier for std of added noise
    :param pca: if True will perform PCA on data and add noise proportional to PCA components.
    :param n_components: Number of components to consider when using PCA.
    :return: Augmented data as a matrix (n_samples x n_features)
    """
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(n_components=n_components)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdMult, size=pca.n_components) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape((n_components, -1))).sum(axis=0)
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:, f] = feat + np.random.normal(scale=stdMult*np.std(feat), size=feat.size)
    return augData


def gen_images(locs, features, len_grid, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., alphaN, beta1, beta2,...), N=n_electrodes
    :param len_grid: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band overall samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    electrodes = locs.shape[0]  # Number of electrodes
    n_samples = features.shape[0]

    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % electrodes == 0
    n_colors = int(features.shape[1] / electrodes)  # color channels of output

    # feat_array_temp = []
    # for c in range(n_colors):
    #     feat_array_temp.append(features[:, c * electrodes: electrodes * (c + 1)])
    features = einops.rearrange(features, 'n (c e) -> c n e', c=n_colors, e=electrodes)

    if augment:
        if pca:
            for c in range(n_colors):
                features[c] = augment_EEG(features[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                features[c] = augment_EEG(features[c], std_mult, pca=False, n_components=n_components)

    # Interpolate the values. np.mgrid: https://www.cnblogs.com/wanghui-garcia/p/10763103.html
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):len_grid * 1j,  # from min to max in x, take len_grid points
                     min(locs[:, 1]):max(locs[:, 1]):len_grid * 1j]    # from min to max in y, take len_grid points
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([n_samples, len_grid, len_grid]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            features[c] = np.append(features[c], np.zeros((n_samples, 4)), axis=1)

    # Interpolating. scipy.interpolate.griddata: https://www.cnblogs.com/ice-coder/p/12652572.html
    for i in range(n_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(points=locs,  # 2D-array, [n, 2], xy coordination
                                               values=features[c][i, :],  # 1DArray, [n,], value of xy
                                               xi=(grid_x, grid_y),  # space to interpolate, usually use numpy.mgrid
                                               method='cubic',  # nearest, linear, cubic
                                               fill_value=np.nan)  # filling when no z
        # print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')

    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]


# def gen_images_original(locs, features, n_gridpoints, normalize=True,
#                augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
#     """
#     Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
#     :param locs: An array with shape [n_electrodes, 2] containing X, Y
#                         coordinates for each electrode.
#     :param features: Feature matrix as [n_samples, n_features]
#                                 Features are as columns.
#                                 Features corresponding to each frequency band are concatenated.
#                                 (alpha1, alpha2, ..., beta1, beta2,...)
#     :param n_gridpoints: Number of pixels in the output images
#     :param normalize:   Flag for whether to normalize each band over all samples
#     :param augment:     Flag for generating augmented images
#     :param pca:         Flag for PCA based data augmentation
#     :param std_mult     Multiplier for std of added noise
#     :param n_components: Number of components in PCA to retain for augmentation
#     :param edgeless:    If True generates edgeless images by adding artificial channels
#                         at four corners of the image with value = 0 (default=False).
#     :return:            Tensor of size [samples, colors, W, H] containing generated
#                         images.
#     """
#     feat_array_temp = []
#     nElectrodes = locs.shape[0]     # Number of electrodes
#
#     # Test whether the feature vector length is divisible by number of electrodes
#     assert features.shape[1] % nElectrodes == 0
#     n_colors = int(features.shape[1] / nElectrodes)
#     for c in range(n_colors):
#         feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
#     if augment:
#         if pca:
#             for c in range(n_colors):
#                 feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
#         else:
#             for c in range(n_colors):
#                 feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
#     n_samples = features.shape[0]
#
#     # Interpolate the values
#     grid_x, grid_y = np.mgrid[
#                      min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
#                      min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
#                      ]
#     temp_interp = []
#     for c in range(n_colors):
#         temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))
#
#     # Generate edgeless images
#     if edgeless:
#         min_x, min_y = np.min(locs, axis=0)
#         max_x, max_y = np.max(locs, axis=0)
#         locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
#         for c in range(n_colors):
#             feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)
#
#     # Interpolating
#     for i in range(n_samples):
#         for c in range(n_colors):
#             temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
#                                                method='cubic', fill_value=np.nan)
#         print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')
#
#     # Normalizing
#     for c in range(n_colors):
#         if normalize:
#             temp_interp[c][~np.isnan(temp_interp[c])] = \
#                 scale(temp_interp[c][~np.isnan(temp_interp[c])])
#         temp_interp[c] = np.nan_to_num(temp_interp[c])
#     return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]
