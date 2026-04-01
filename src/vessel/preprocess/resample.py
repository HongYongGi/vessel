"""Resample medical images to a uniform target spacing."""

from __future__ import annotations

import SimpleITK as sitk


def resample_image(
    image: sitk.Image,
    target_spacing: list[float],
    is_label: bool = False,
) -> sitk.Image:
    """Resample a SimpleITK image to *target_spacing*.

    Parameters
    ----------
    image : sitk.Image
        Input image to resample.
    target_spacing : list[float]
        Desired voxel spacing in each dimension, e.g. ``[1.0, 1.0, 1.0]``.
    is_label : bool
        If ``True``, use nearest-neighbour interpolation and a default pixel
        value of ``0``.  Otherwise use B-spline interpolation with a default
        pixel value of ``-1024`` (air in Hounsfield units).

    Returns
    -------
    sitk.Image
        Resampled image with the requested spacing.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Compute new size so that the physical extent stays the same
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputPixelType(image.GetPixelID())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(-1024)

    return resampler.Execute(image)
