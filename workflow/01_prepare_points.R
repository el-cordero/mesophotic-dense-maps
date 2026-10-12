#!/usr/bin/env Rscript
# Step 1. Write one point file per tile from the CPCe annotations.
#
# Inputs (not distributed, available on request)
#   --raw-points   point table written by pointcoral from the CPCe files, one row per point
#                  with image_name, point_index and the harmonized class (class_id, ml_class)
#   --points-dir   full-image coordinate files, one "x y class" line per CPCe point in CPCe order
#   --tiles-csv    tile layout (data/tiles.csv by default), with the crop window of every tile
#
# Labels are harmonized with the pointcoral crosswalk, the excluded categories listed below are
# dropped, and each retained point is written to the tile that contains it in tile pixel
# coordinates (1-based, origin at the upper-left corner). Output goes to <tiles-dir>/points/{split}/
# and audit tables to outputs/point_prep/.
#
# Example
#   Rscript workflow/01_prepare_points.R --raw-points pointcoral_total_raw_points.csv \
#     --points-dir points_txt --tiles-dir tiles

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(readr)
  library(stringr)
  library(tibble)
  library(pointcoral)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0L) return(default)
  args[[hit + 1L]]
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else file.path("workflow", "01_prepare_points.R")
project_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)

tiles_csv <- normalizePath(get_arg("--tiles-csv", file.path(project_root, "data", "tiles.csv")), mustWork = TRUE)
out_dataset <- normalizePath(get_arg("--tiles-dir", file.path(project_root, "tiles")), mustWork = FALSE)
pointcoral_raw_points <- normalizePath(get_arg("--raw-points"), mustWork = TRUE)
source_points_dir <- normalizePath(get_arg("--points-dir"), mustWork = TRUE)
report_dir <- get_arg("--report-dir", file.path(project_root, "outputs", "point_prep"))

excluded_ml_classes <- c(
  "ZOANTHIDS (Z)",
  "DISEASED CORALS (DC)",
  "OTHER LIVE (OL)",
  "DEAD CORAL WITH ALGAE (DCA)",
  "ASCIDIAN",
  "TAPE, WAND, SHADOW (TWS)",
  "UNKNOWNS (U)"
)
excluded_class_ids <- c(3L, 9L, 10L, 11L, 12L, 13L, 14L)

class_colors <- tribble(
  ~class_id, ~color_hex,
  0L, "#ff4d4d",
  1L, "#ff944d",
  2L, "#3399ff",
  4L, "#33cc66",
  5L, "#cc3366",
  6L, "#7fbf3f",
  7L, "#8b0000",
  8L, "#d8c28a"
)

pointcoral_class_lookup <- read_label_crosswalk(system.file("extdata", "pointcoral_example_crosswalk.csv", package = "pointcoral")) |>
  distinct(.data$class_id, .data$ml_class) |>
  filter(
    !.data$ml_class %in% excluded_ml_classes,
    !.data$class_id %in% excluded_class_ids
  ) |>
  arrange(.data$class_id) |>
  left_join(class_colors, by = "class_id") |>
  mutate(color_hex = coalesce(.data$color_hex, "#bdbdbd"))

manifest <- read_csv(tiles_csv, show_col_types = FALSE) |>
  mutate(
    image_stem = tools::file_path_sans_ext(.data$image_file),
    tile_stem = .data$tile,
    points_basename = .data$points_file
  )

dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

raw_labels_all <- read_csv(pointcoral_raw_points, show_col_types = FALSE) |>
  mutate(
    image_stem = as.character(.data$image_name),
    point_index = as.integer(.data$point_index),
    pointcoral_class_id = as.integer(.data$class_id),
    pointcoral_ml_class = as.character(.data$ml_class),
    pointcoral_major_category = as.character(.data$major_category),
    excluded_from_ml = .data$pointcoral_ml_class %in% excluded_ml_classes |
      .data$pointcoral_class_id %in% excluded_class_ids
  )

raw_labels <- raw_labels_all |>
  filter(.data$image_stem %in% unique(manifest$image_stem)) |>
  arrange(.data$image_stem, .data$point_index, .data$source_file) |>
  distinct(.data$image_stem, .data$point_index, .keep_all = TRUE)

raw_duplicates <- raw_labels_all |>
  filter(.data$image_stem %in% unique(manifest$image_stem)) |>
  count(.data$image_stem, .data$point_index, name = "n") |>
  filter(.data$n > 1L)
write_csv(raw_duplicates, file.path(report_dir, "duplicate_raw_label_keys.csv"))

write_csv(
  raw_labels |>
    count(
      .data$pointcoral_class_id,
      .data$pointcoral_ml_class,
      .data$pointcoral_major_category,
      .data$excluded_from_ml,
      name = "points"
    ) |>
    arrange(.data$pointcoral_class_id, .data$pointcoral_ml_class),
  file.path(report_dir, "pointcoral_label_mapping_audit.csv")
)
write_csv(
  raw_labels |> filter(.data$excluded_from_ml),
  file.path(report_dir, "pointcoral_excluded_points.csv")
)

image_units <- manifest |>
  distinct(
    .data$split,
    .data$image_stem,
    .data$points_basename,
    .data$original_width,
    .data$original_height
  ) |>
  arrange(.data$split, .data$image_stem)

parse_point_file <- function(path, image_stem, split, points_basename, original_width, original_height) {
  if (!file.exists(path)) {
    return(tibble(
      split = split,
      image_stem = image_stem,
      points_basename = points_basename,
      point_index = integer(),
      x_px = integer(),
      y_px = integer(),
      old_class_id = integer(),
      original_width = integer(),
      original_height = integer(),
      source_point_path = character(),
      malformed = logical()
    ))
  }

  lines <- readLines(path, warn = FALSE)
  if (length(lines) == 0L) {
    return(tibble(
      split = split,
      image_stem = image_stem,
      points_basename = points_basename,
      point_index = integer(),
      x_px = integer(),
      y_px = integer(),
      old_class_id = integer(),
      original_width = integer(),
      original_height = integer(),
      source_point_path = character(),
      malformed = logical()
    ))
  }

  parsed <- str_split(str_trim(lines), "\\s+", simplify = TRUE)
  malformed <- ncol(parsed) != 3L
  if (malformed) {
    parsed <- matrix(NA_character_, nrow = length(lines), ncol = 3)
  }

  tibble(
    split = split,
    image_stem = image_stem,
    points_basename = points_basename,
    point_index = seq_along(lines),
    x_px = suppressWarnings(as.integer(as.numeric(parsed[, 1]))),
    y_px = suppressWarnings(as.integer(as.numeric(parsed[, 2]))),
    old_class_id = suppressWarnings(as.integer(as.numeric(parsed[, 3]))),
    original_width = as.integer(original_width),
    original_height = as.integer(original_height),
    source_point_path = path,
    malformed = malformed
  )
}

coord_points <- pmap_dfr(
  image_units,
  function(split, image_stem, points_basename, original_width, original_height) {
    parse_point_file(
      file.path(source_points_dir, points_basename),
      image_stem,
      split,
      points_basename,
      original_width,
      original_height
    )
  }
)

missing_coord_files <- image_units |>
  mutate(source_point_path = file.path(source_points_dir, .data$points_basename)) |>
  filter(!file.exists(.data$source_point_path))
write_csv(missing_coord_files, file.path(report_dir, "missing_full_image_coordinate_files.csv"))

full_points <- coord_points |>
  left_join(
    raw_labels |>
      select(
        image_stem,
        point_index,
        source_file,
        source_path,
        raw_code,
        raw_label,
        full_label,
        clean_label,
        label_class,
        cpce_major_category,
        pointcoral_major_category,
        pointcoral_ml_class,
        pointcoral_class_id,
        excluded_from_ml
      ),
    by = c("image_stem", "point_index"),
    relationship = "many-to-one"
  ) |>
  mutate(
    label_missing = is.na(.data$pointcoral_ml_class),
    excluded_from_ml = coalesce(.data$excluded_from_ml, TRUE),
    export_class_id = .data$pointcoral_class_id,
    export_ml_class = .data$pointcoral_ml_class
  )

write_csv(
  full_points |>
    group_by(.data$split, .data$image_stem, .data$points_basename) |>
    summarise(
      n_coordinate_points = n(),
      n_missing_labels = sum(.data$label_missing),
      n_excluded_points = sum(.data$excluded_from_ml, na.rm = TRUE),
      n_malformed_coordinate_rows = sum(.data$malformed),
      n_coordinate_out_of_range = sum(
        is.na(.data$x_px) | is.na(.data$y_px) |
          .data$x_px < 1L | .data$y_px < 1L |
          .data$x_px > .data$original_width | .data$y_px > .data$original_height
      ),
      .groups = "drop"
    ) |>
    arrange(desc(.data$n_missing_labels), desc(.data$n_excluded_points), .data$image_stem),
  file.path(report_dir, "pointcoral_image_join_audit.csv")
)

write_csv(
  full_points |>
    filter(.data$label_missing | .data$malformed | is.na(.data$x_px) | is.na(.data$y_px)),
  file.path(report_dir, "pointcoral_unresolved_points.csv")
)
write_csv(full_points, file.path(report_dir, "pointcoral_full_image_points.csv"))

points_root <- file.path(out_dataset, "points")
if (dir.exists(points_root)) {
  unlink(file.path(points_root, c("train", "val", "test")), recursive = TRUE, force = TRUE)
}
dir.create(points_root, recursive = TRUE, showWarnings = FALSE)

tile_points_rows <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  tile <- manifest[i, ]
  df <- full_points |>
    filter(
      .data$image_stem == tile$image_stem,
      !.data$excluded_from_ml,
      !.data$label_missing,
      !is.na(.data$x_px),
      !is.na(.data$y_px),
      .data$x_px >= tile$x0,
      .data$x_px <= tile$x1,
      .data$y_px >= tile$y0,
      .data$y_px <= tile$y1
    ) |>
    transmute(
      split = tile$split,
      tile_stem = tile$tile_stem,
      image_stem = tile$image_stem,
      point_index = .data$point_index,
      x_px = as.integer(.data$x_px - tile$x0 + 1L),
      y_px = as.integer(.data$y_px - tile$y0 + 1L),
      class_id = as.integer(.data$export_class_id),
      ml_class = .data$export_ml_class,
      raw_code = .data$raw_code,
      old_class_id = .data$old_class_id
    ) |>
    arrange(.data$point_index)

  split_dir <- file.path(points_root, tile$split)
  dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(split_dir, paste0(tile$tile_stem, ".txt"))
  write_lines(sprintf("%d %d %d", df$x_px, df$y_px, df$class_id), out_path)
  tile_points_rows[[i]] <- df
}

tile_points <- bind_rows(tile_points_rows)

write_csv(
  pointcoral_class_lookup |>
    transmute(class_id, class_name = ml_class, color_hex),
  file.path(out_dataset, "class_lookup.csv")
)
write_csv(
  pointcoral_class_lookup |>
    transmute(class_id, class_name = ml_class, color_hex),
  file.path(report_dir, "pointcoral_class_lookup.csv")
)
write_csv(tile_points, file.path(report_dir, "pointcoral_tile_points.csv"))
write_csv(
  tile_points |>
    count(.data$split, .data$class_id, .data$ml_class, name = "points") |>
    arrange(.data$split, .data$class_id),
  file.path(report_dir, "pointcoral_class_counts_by_split.csv")
)

tile_count_audit <- manifest |>
  transmute(
    split = .data$split,
    tile_stem = .data$tile_stem,
    manifest_n_points_tile = .data$n_points_tile
  ) |>
  left_join(
    tile_points |> count(.data$split, .data$tile_stem, name = "rebuilt_n_points_tile"),
    by = c("split", "tile_stem"),
    relationship = "one-to-one"
  ) |>
  mutate(
    rebuilt_n_points_tile = coalesce(.data$rebuilt_n_points_tile, 0L),
    point_count_delta = .data$rebuilt_n_points_tile - .data$manifest_n_points_tile
  )
write_csv(tile_count_audit, file.path(report_dir, "pointcoral_tile_count_audit.csv"))

cat("Wrote pointcoral-CSV-refactored points to: ", out_dataset, "\n", sep = "")
cat("Coordinate source: ", source_points_dir, "\n", sep = "")
cat("Pointcoral label source: ", pointcoral_raw_points, "\n", sep = "")
cat("Excluded ML classes: ", paste(excluded_ml_classes, collapse = "; "), "\n", sep = "")
cat("Reports: ", report_dir, "\n", sep = "")
