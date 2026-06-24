import numpy as np
from PIL import Image
from src.stereoimage_generation import apply_stereo_divergence


def create_quilt(original_image, depthmap, cols, rows, divergence,
                 stereo_offset_exponent=1.0, fill_technique='polylines_sharp', rotate=False, focus=0.5):
    """Creates a Looking Glass quilt image.

    Generates cols*rows views at evenly-spaced angles from +divergence (leftmost camera)
    to -divergence (rightmost camera), arranged per Looking Glass convention:
    left-to-right within each row, rows stacked bottom-to-top in the output image.

    :param original_image: PIL Image or numpy array
    :param depthmap: normalized depth array (white=near, black=far)
    :param int cols: number of view columns
    :param int rows: number of view rows
    :param float divergence: total angle sweep in percent of image width; reuses STEREO_DIVERGENCE
    :param float stereo_offset_exponent: depth power curve; reuses STEREO_OFFSET_EXPONENT
    :param str fill_technique: gap-fill method; reuses STEREO_FILL_ALGO
    :param bool rotate: rotate each view 90° counter-clockwise before assembling the grid
    :param float focus: normalized depth (0=far, 1=near) that appears at zero parallax (screen surface)
    """
    original_image = np.asarray(original_image)
    total_views = cols * rows

    views = []
    for i in range(total_views):
        t = i / (total_views - 1) if total_views > 1 else 0.5
        # +divergence = leftmost camera (view 0), -divergence = rightmost camera (last view)
        angle = divergence * (1.0 - 2.0 * t)

        if abs(angle) < 1e-6:
            view = original_image.copy()
        else:
            view = apply_stereo_divergence(
                original_image, depthmap, angle, 0.0, stereo_offset_exponent, fill_technique, focus=focus
            )
        if rotate:
            view = np.rot90(view)
        views.append(view)

    # When rotated, swap grid dimensions so the quilt is in landscape orientation.
    out_cols = rows if rotate else cols
    out_rows = cols if rotate else rows

    # Build rows left-to-right, then stack bottom-to-top (Looking Glass convention).
    # numpy is top-to-bottom, so reverse row order before vstacking.
    row_strips = []
    for r in range(out_rows):
        row_strips.append(np.hstack([views[r * out_cols + c] for c in range(out_cols)]))
    quilt = np.vstack(row_strips[::-1])

    return Image.fromarray(quilt)
