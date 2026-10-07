# Input and output formats

## Synthetic settings manifest

Required columns:

```csv
image_id,level,target_width_px,seed
```

`image_id` must be a unique PNG basename. `target_width_px` is a positive nominal width, `seed` is an integer, and `level` is a generation-group label.

## Synthetic design references

The generator writes these columns to `groundtruth.csv`:

```csv
image_id,level,target_width_px,gt_max_width_px,gt_max_width_mm,gt_center_x,gt_center_y
```

The `gt_` columns describe the **unperturbed design reference**, not an independently measured maximum of the final perturbed mask. Coordinates refer to the original image in pixels.

## Co-located result coordinates

The output contains `reference_center_x`, `reference_center_y`, `center_x`, `center_y`, and `center_mode`. Coordinates are expressed in pixels in the original image frame. Width results are expressed in millimeters.
