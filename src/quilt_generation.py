import numpy as np
from PIL import Image


def create_quilt(original_image, depthmap, cols, rows, divergence,
                 stereo_offset_exponent=1.0, fill_technique='polylines_sharp', rotate=False, focus=0.5):
    """Creates a Looking Glass quilt image.

    Generates cols*rows views at evenly-spaced angles from +divergence (leftmost camera)
    to -divergence (rightmost camera), arranged per Looking Glass convention:
    left-to-right within each row, rows stacked bottom-to-top in the output image.

    Uses a vectorized inverse warp: all views are computed in a single NumPy
    gather operation rather than one serial loop per view, so fill_technique
    is not needed (every destination pixel is directly sampled — no gaps).

    :param original_image: PIL Image or numpy array
    :param depthmap: normalized depth array (white=near, black=far)
    :param int cols: number of view columns
    :param int rows: number of view rows
    :param float divergence: total angle sweep in percent of image width
    :param float stereo_offset_exponent: depth power curve (ignored when focus != 0)
    :param str fill_technique: accepted for API compatibility, not used
    :param bool rotate: rotate source 90° before generating views (landscape layout)
    :param float focus: normalized depth (0=far, 1=near) at zero parallax
    """
    original_image = np.asarray(original_image)
    # Rotate source and depth before view generation so parallax stays horizontal.
    if rotate:
        original_image = np.rot90(original_image)
        depthmap = np.rot90(depthmap)

    H, W, C = original_image.shape
    total_views = cols * rows

    # Normalize depth once for all views.
    depth_min, depth_max = float(depthmap.min()), float(depthmap.max())
    if depth_max > depth_min:
        depth_norm = (depthmap.astype(np.float32) - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros((H, W), dtype=np.float32)

    # Apply focus offset (shifts zero-parallax plane) or exponent curve.
    if focus != 0.0:
        depth_for_shift = depth_norm - focus          # can be negative for far objects
    else:
        depth_for_shift = depth_norm ** stereo_offset_exponent

    # View angles: +divergence (leftmost, view 0) to -divergence (rightmost, last view).
    t = np.linspace(0.0, 1.0, total_views, dtype=np.float32) if total_views > 1 \
        else np.array([0.5], dtype=np.float32)
    angles = divergence * (1.0 - 2.0 * t)  # (V,)

    # Vectorized inverse warp:
    # For each view v and destination column w, the source column is:
    #   col_s = w - angle[v] * depth[h, w] * W / 100
    # Shape: (V, H, W)
    col_s = np.clip(
        np.arange(W, dtype=np.float32)[np.newaxis, np.newaxis, :] -
        angles[:, np.newaxis, np.newaxis] * depth_for_shift[np.newaxis, :, :] * (W / 100.0),
        0, W - 1
    ).astype(np.int32)

    # Gather all views in one advanced-indexing call.
    # row_idx (1, H, 1) broadcasts with col_s (V, H, W) → result (V, H, W, C).
    row_idx = np.arange(H, dtype=np.int32)[np.newaxis, :, np.newaxis]
    views_array = original_image[row_idx, col_s]  # (V, H, W, C)

    # Assemble quilt grid. Rotate swaps grid dimensions for landscape layout.
    out_cols = rows if rotate else cols
    out_rows = cols if rotate else rows

    # reshape → transpose → reshape turns (V,H,W,C) into row strips,
    # then [::-1] reverses to Looking Glass bottom-to-top row order.
    quilt = (views_array
             .reshape(out_rows, out_cols, H, W, C)
             .transpose(0, 2, 1, 3, 4)               # (out_rows, H, out_cols, W, C)
             .reshape(out_rows, H, out_cols * W, C)
             [::-1]                                   # bottom row first in output image
             .copy()                                  # contiguous memory for final reshape
             .reshape(out_rows * H, out_cols * W, C))

    return Image.fromarray(quilt)
