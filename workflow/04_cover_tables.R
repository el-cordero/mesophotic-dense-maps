#!/usr/bin/env Rscript
# Step 4. Point-count cover and dense cover for every tile and photoquadrat.
#
# For each tile, point-count cover is the share of CPCe points in each class and dense cover
# is the share of predicted pixels in each class, computed at confidence thresholds of
# 0.1 to 1.0. Tiles are pooled to photoquadrats by summing counts. The photoquadrat table,
# which the analysis in step 6 reads, is written to data/photoquadrat_cover_by_threshold.csv.gz
# and the intermediate tables to outputs/cover_tables/.
#
# Example
#   Rscript workflow/04_cover_tables.R --tiles-dir tiles --output-dir outputs

suppressPackageStartupMessages({
  library(terra)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- sub("^--file=", "", script_arg[1] %||% file.path("workflow", "04_cover_tables.R"))
project_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] == length(args)) return(default)
  args[hit[length(hit)] + 1]
}

dataset_root <- normalizePath(arg_value("--tiles-dir", file.path(project_root, "tiles")), mustWork = TRUE)
output_root <- normalizePath(arg_value("--output-dir", file.path(project_root, "outputs")), mustWork = TRUE)
table_dir <- file.path(output_root, "cover_tables")
data_dir <- file.path(project_root, "data")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
recompute_ml_cover <- "--recompute-ml-cover" %in% args


thresholds <- seq(0.10, 1.00, by = 0.10)
splits <- c("train", "val", "test")
image_exts <- c(".jpg", ".jpeg", ".png", ".tif", ".tiff")

classes <- read_csv(file.path(project_root, "config/classes.csv"), show_col_types = FALSE) %>%
  mutate(class_id = as.integer(class_id))
class_ids <- classes$class_id
class_names <- setNames(classes$class_name, classes$class_id)
class_cols <- setNames(classes$color_hex, classes$class_id)

parse_base_stem <- function(stem) {
  str_replace(stem, "_tile_r\\d+_c\\d+$", "")
}

find_image <- function(split, stem) {
  image_dir <- file.path(dataset_root, "images", split)
  candidates <- file.path(image_dir, paste0(stem, image_exts))
  candidates[file.exists(candidates)][1] %||% NA_character_
}

read_points <- function(path) {
  if (!file.exists(path) || file.info(path)$size == 0) {
    return(tibble(x = integer(), y = integer(), class_id = integer()))
  }
  read_table(
    path,
    col_names = c("x", "y", "class_id"),
    col_types = cols(x = col_double(), y = col_double(), class_id = col_integer()),
    progress = FALSE
  ) %>%
    mutate(x = as.integer(round(x)), y = as.integer(round(y)))
}

prediction_files <- bind_rows(lapply(splits, function(split) {
  pred_dir <- file.path(output_root, "predictions", split)
  files <- list.files(pred_dir, pattern = "\\.png$", full.names = TRUE)
  tibble(split = split, pred_path = files, stem = tools::file_path_sans_ext(basename(files)))
})) %>%
  mutate(
    image_path = vapply(seq_along(stem), function(i) find_image(split[i], stem[i]), character(1)),
    point_path = file.path(dataset_root, "points", split, paste0(stem, ".txt")),
    confidence_path = file.path(output_root, "confidence", split, paste0(stem, "_confidence.png")),
    base_stem = parse_base_stem(stem)
  ) %>%
  filter(!is.na(image_path), file.exists(point_path), file.exists(confidence_path))

write_csv(prediction_files, file.path(table_dir, "analysis_file_index.csv"))

if (nrow(prediction_files) == 0) {
  stop("No matching prediction/image/point/confidence files found.")
}

message("Files indexed: ", nrow(prediction_files))

point_cover_tile <- bind_rows(lapply(seq_len(nrow(prediction_files)), function(i) {
  row <- prediction_files[i, ]
  pts <- read_points(row$point_path)
  total <- nrow(pts)
  tibble(class_id = class_ids) %>%
    left_join(count(pts, class_id, name = "point_count"), by = "class_id") %>%
    mutate(
      point_count = replace_na(point_count, 0L),
      point_total = total,
      point_cover_pct = ifelse(total > 0, point_count / total * 100, NA_real_),
      split = row$split,
      stem = row$stem,
      base_stem = row$base_stem
    )
}))

write_csv(point_cover_tile, file.path(table_dir, "point_cover_by_tile.csv"))

point_cover_photoquadrat <- point_cover_tile %>%
  group_by(split, base_stem, class_id) %>%
  summarise(point_count = sum(point_count), point_total = sum(first(point_total), na.rm = TRUE), .groups = "drop_last") %>%
  ungroup() %>%
  group_by(split, base_stem) %>%
  mutate(point_total = sum(point_count), point_cover_pct = ifelse(point_total > 0, point_count / point_total * 100, NA_real_)) %>%
  ungroup()

write_csv(point_cover_photoquadrat, file.path(table_dir, "point_cover_by_photoquadrat.csv"))

mask_frequency_for_threshold <- function(mask_path, conf_path, threshold) {
  mask <- rast(mask_path)
  conf <- rast(conf_path)
  if (!compareGeom(mask, conf, stopOnError = FALSE)) {
    conf <- resample(conf, mask, method = "near")
  }
  total_pixels <- ncell(mask)
  conf_cut <- threshold * 255
  keep <- conf >= conf_cut
  confident_pixels <- as.numeric(global(keep, "sum", na.rm = TRUE)[1, 1])
  masked <- mask
  masked[!keep] <- NA
  freq_tbl <- as.data.frame(freq(masked, digits = 0))
  if (nrow(freq_tbl) == 0) {
    freq_tbl <- tibble(value = integer(), count = integer())
  }
  count_col <- if ("count" %in% names(freq_tbl)) "count" else "counts"
  value_col <- if ("value" %in% names(freq_tbl)) "value" else "layer"
  freq_tbl <- freq_tbl[!is.na(freq_tbl[[value_col]]), , drop = FALSE]
  out <- tibble(class_id = class_ids) %>%
    left_join(
      tibble(
        class_id = as.integer(freq_tbl[[value_col]]),
        ml_pixel_count = as.numeric(freq_tbl[[count_col]])
      ),
      by = "class_id"
    ) %>%
    mutate(
      ml_pixel_count = replace_na(ml_pixel_count, 0),
      threshold = threshold,
      total_pixels = total_pixels,
      confident_pixels = confident_pixels,
      confident_pct = ifelse(total_pixels > 0, confident_pixels / total_pixels * 100, NA_real_),
      ml_cover_confident_pct = ifelse(confident_pixels > 0, ml_pixel_count / confident_pixels * 100, NA_real_),
      ml_cover_total_pct = ifelse(total_pixels > 0, ml_pixel_count / total_pixels * 100, NA_real_)
    )
  out
}

ml_cover_tile_path <- file.path(table_dir, "ml_cover_by_tile_threshold.csv")
if (file.exists(ml_cover_tile_path) && !recompute_ml_cover) {
  message("Using cached ML cover table: ", ml_cover_tile_path)
  ml_cover_tile <- read_csv(ml_cover_tile_path, show_col_types = FALSE)
} else {
  message("Computing ML cover by confidence threshold...")
  ml_cover_tile <- bind_rows(lapply(seq_len(nrow(prediction_files)), function(i) {
    if (i %% 100 == 0) message("  ", i, " / ", nrow(prediction_files))
    row <- prediction_files[i, ]
    bind_rows(lapply(thresholds, function(thr) {
      mask_frequency_for_threshold(row$pred_path, row$confidence_path, thr)
    })) %>%
      mutate(split = row$split, stem = row$stem, base_stem = row$base_stem)
  }))

  write_csv(ml_cover_tile, ml_cover_tile_path)
}

photoquadrat_pixel_totals <- ml_cover_tile %>%
  distinct(split, base_stem, stem, threshold, total_pixels, confident_pixels) %>%
  group_by(split, base_stem, threshold) %>%
  summarise(
    total_pixels = sum(total_pixels),
    confident_pixels = sum(confident_pixels),
    .groups = "drop"
  )

ml_cover_photoquadrat <- ml_cover_tile %>%
  group_by(split, base_stem, threshold, class_id) %>%
  summarise(
    ml_pixel_count = sum(ml_pixel_count),
    .groups = "drop"
  ) %>%
  left_join(photoquadrat_pixel_totals, by = c("split", "base_stem", "threshold")) %>%
  group_by(split, base_stem, threshold) %>%
  mutate(
    confident_total = sum(ml_pixel_count),
    confident_pct = ifelse(first(total_pixels) > 0, first(confident_pixels) / first(total_pixels) * 100, NA_real_),
    ml_cover_confident_pct = ifelse(confident_total > 0, ml_pixel_count / confident_total * 100, NA_real_),
    ml_cover_total_pct = ifelse(first(total_pixels) > 0, ml_pixel_count / first(total_pixels) * 100, NA_real_)
  ) %>%
  ungroup() %>%
  select(-confident_total)

write_csv(ml_cover_photoquadrat, file.path(table_dir, "ml_cover_by_photoquadrat_threshold.csv"))

comparison_tile <- ml_cover_tile %>%
  left_join(
    point_cover_tile %>% select(split, stem, base_stem, class_id, point_count, point_total, point_cover_pct),
    by = c("split", "stem", "base_stem", "class_id")
  ) %>%
  mutate(
    class_name = class_names[as.character(class_id)],
    error_confident_pct = ml_cover_confident_pct - point_cover_pct,
    abs_error_confident_pct = abs(error_confident_pct),
    sq_error_confident_pct = error_confident_pct^2,
    error_total_pct = ml_cover_total_pct - point_cover_pct
  )

write_csv(comparison_tile, file.path(table_dir, "tile_point_vs_ml_cover_by_threshold.csv"))

comparison_photo <- ml_cover_photoquadrat %>%
  left_join(
    point_cover_photoquadrat %>% select(split, base_stem, class_id, point_count, point_total, point_cover_pct),
    by = c("split", "base_stem", "class_id")
  ) %>%
  mutate(
    class_name = class_names[as.character(class_id)],
    error_confident_pct = ml_cover_confident_pct - point_cover_pct,
    abs_error_confident_pct = abs(error_confident_pct),
    sq_error_confident_pct = error_confident_pct^2,
    error_total_pct = ml_cover_total_pct - point_cover_pct
  )

write_csv(comparison_photo, file.path(table_dir, "photoquadrat_point_vs_ml_cover_by_threshold.csv"))
write_csv(comparison_photo, file.path(data_dir, "photoquadrat_cover_by_threshold.csv.gz"))

message("Photoquadrat cover table: ", file.path(data_dir, "photoquadrat_cover_by_threshold.csv.gz"))
