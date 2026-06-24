import numpy as np
from PIL import Image


def create_quilt(original_image, depthmap, cols, rows, divergence,
                 stereo_offset_exponent=1.0, fill_technique='polylines_sharp', rotate=False, focus=0.5,
                 background_plate=None):
    """Creates a Looking Glass quilt image.

    Generates cols*rows views at evenly-spaced angles from +divergence (leftmost camera)
    to -divergence (rightmost camera), arranged per Looking Glass convention:
    left-to-right within each row, rows stacked bottom-to-top in the output image.

    Uses a forward warp (scatter): each source pixel is placed at its correct
    destination, with far pixels processed first so near pixels overwrite them.
    Gaps are filled with a vectorized nearest-neighbour pass (no Python loops
    over pixels). The sort is computed once and reused across all views.

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
    # Resolve background plate before rotation so both are rotated together.
    fill_source = np.asarray(background_plate) if background_plate is not None else original_image
    if rotate:
        original_image = np.rot90(original_image)
        depthmap = np.rot90(depthmap)
        fill_source = np.rot90(fill_source)

    H, W, C = original_image.shape
    total_views = cols * rows

    # Normalize depth once for all views.
    depth_min, depth_max = float(depthmap.min()), float(depthmap.max())
    if depth_max > depth_min:
        depth_norm = (depthmap.astype(np.float32) - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros((H, W), dtype=np.float32)

    if focus != 0.0:
        depth_for_shift = depth_norm - focus
    else:
        depth_for_shift = depth_norm ** stereo_offset_exponent

    # View angles: +divergence (leftmost, view 0) to -divergence (rightmost, last).
    t = np.linspace(0.0, 1.0, total_views, dtype=np.float32) if total_views > 1 \
        else np.array([0.5], dtype=np.float32)
    angles = divergence * (1.0 - 2.0 * t)  # (V,)

    # Sort pixels by depth ascending (far first) once — reused for every view.
    # Near pixels are written last and therefore overwrite far pixels (correct occlusion).
    row_idx = np.arange(H, dtype=np.int32)[:, np.newaxis]  # (H, 1)
    col_range = np.arange(W, dtype=np.int32)               # (W,)
    order = np.argsort(depth_for_shift, axis=1)            # (H, W) ascending depth
    depth_sorted = depth_for_shift[row_idx, order]         # (H, W)
    source_sorted = original_image[row_idx, order]         # (H, W, C)
    order_f = order.astype(np.float32)
    W_scale = W / 100.0

    # Assemble quilt grid. Rotate swaps grid dimensions for landscape layout.
    out_cols = rows if rotate else cols
    out_rows = cols if rotate else rows

    # Pre-allocate once; write each view directly into its grid slot.
    # This avoids accumulating a views list and the costly stack+transpose+copy chain.
    quilt = np.zeros((out_rows * H, out_cols * W, C), dtype=np.uint8)

    for v in range(total_views):
        a = float(angles[v])

        # Forward warp: destination column for each depth-sorted source pixel.
        col_dst = np.clip(
            order_f + a * depth_sorted * W_scale,
            0, W - 1
        ).astype(np.int32)  # (H, W)

        # LG convention: view 0 is leftmost, placed at bottom-left of the quilt.
        # Views increase left→right within a row, then bottom→top across rows.
        col_in_grid = v % out_cols
        row_in_grid = v // out_cols
        out_row = out_rows - 1 - row_in_grid  # row 0 sits at the image bottom

        qs = quilt[out_row * H:(out_row + 1) * H, col_in_grid * W:(col_in_grid + 1) * W]

        # Fill gaps from background plate first; scatter foreground on top.
        # Near pixels were sorted last so they overwrite far pixels (correct occlusion).
        qs[:] = fill_source
        qs[row_idx, col_dst] = source_sorted

    return Image.fromarray(quilt)
