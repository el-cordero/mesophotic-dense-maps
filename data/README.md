# Shared data

These tables are derived from the photoquadrat images, the CPCe point annotations and the saved prediction rasters. They hold everything the analysis (step 6) and most figures (step 7) need. The images, CPCe files and rasters themselves are available from the corresponding author on reasonable request. The tables are released under CC BY 4.0.

Photoquadrats are identified by `<date>_<site>_<depth>_meters_<image name>` (for example `01_Jun_2009_Hole_in_the_Wall_48_meters_HIW_158_W_U-1`) and tiles add `_tile_r<row>_c<column>`. Site names follow the original survey folders, so El Hoyo appears as `Hoyo Terrace` (and as `Hoyo Terrrace` in some file names).

## tiles.csv

One row per tile (3198 tiles from 533 photoquadrats).

| Column | Meaning |
|---|---|
| `tile`, `photoquadrat` | tile and parent photoquadrat identifiers |
| `split` | `train`, `val` or `test` |
| `survey` | survey unit (site, date and depth) used to assign partitions |
| `date_folder`, `site_folder`, `depth_folder` | survey folders of the original images |
| `image_file`, `points_file` | original image file and its full-image point file |
| `tile_row`, `tile_col` | position of the tile in the 2 x 3 grid |
| `x0`, `y0`, `x1`, `y1` | crop window in the original image (1-based pixels, inclusive) |
| `tile_width`, `tile_height`, `original_width`, `original_height` | sizes in pixels |
| `n_points_tile` | retained CPCe points in the tile |

## photoquadrats.csv

One row per photoquadrat with `split`, `site`, `depth_m`, `date`, `n_tiles` and `n_points`. Written by step 5.

## point_predictions.csv.gz

One row per retained CPCe point (53,271 points). Written by step 5.

| Column | Meaning |
|---|---|
| `split`, `tile`, `photoquadrat` | partition and identifiers |
| `x`, `y` | point position in the tile (1-based pixels, origin at the upper-left corner) |
| `label_class_id` | analyst label (class IDs in `config/classes.csv`) |
| `predicted_class_id` | class of the dense map at the point |
| `confidence` | maximum softmax probability at the point, read from the 8-bit confidence raster |
| `correct` | whether the predicted class matches the label |
| `tile_width`, `tile_height` | raster size in pixels |

## photoquadrat_cover_by_threshold.csv.gz

Point-count cover and dense cover for every photoquadrat, class and confidence threshold from 0.1 to 1.0 (533 photoquadrats x 8 classes x 10 thresholds). Written by step 4. The photoquadrat identifier is in `base_stem`.

| Column | Meaning |
|---|---|
| `threshold` | confidence threshold |
| `class_id`, `class_name` | benthic class |
| `point_count`, `point_total`, `point_cover_pct` | CPCe points in the class, all points, and point-count cover (%) |
| `ml_pixel_count` | predicted pixels of the class at or above the threshold |
| `total_pixels`, `confident_pixels`, `confident_pct` | all pixels, pixels at or above the threshold, and their share (%) |
| `ml_cover_confident_pct` | dense cover (%) among retained pixels |
| `ml_cover_total_pct` | dense cover (%) over the whole photoquadrat |
| `error_*` | dense minus point-count cover (pp), with absolute and squared values |
