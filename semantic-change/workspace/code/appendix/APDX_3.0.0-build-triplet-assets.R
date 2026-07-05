suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(ggalluvial)
  library(rcartocolor)
  library(scales)
  library(grid)
  library(grDevices)
  library(paletteer)
})

# Experimental alluvial prep for RDF triplet outputs
#
# This script is designed for interactive use after a triplet run finishes.
# It assumes the structured outputs created by:
#   APDX_1.0.0-rdf-triplet-extraction-api.ipynb
#
# Main products:
#   p_pairs
#     alluvial plot of top subject-object links
#   p_dark_matter_over_time
#     stacked count plot of what dark matter is linked to over time

script_path <- if (!is.null(sys.frames()[[1]]$ofile)) {
  normalizePath(sys.frames()[[1]]$ofile, winslash = "/", mustWork = TRUE)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)[1]
  if (!is.na(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg), winslash = "/", mustWork = TRUE)
  } else {
    normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  }
}
script_dir <- if (file.info(script_path)$isdir) script_path else dirname(script_path)
workspace_helper <- normalizePath(
  file.path(script_dir, "..", "..", "..", "..", "shared-assets", "code", "workspace_rooting", "workspace_paths.R"),
  winslash = "/",
  mustWork = TRUE
)
source(workspace_helper)

paths <- canonical_workspace_paths(file.path(script_dir, "..", ".."))
SCRIPT_DIR <- file.path(paths$code, "appendix")
RUNS_DIR <- file.path(paths$outputs, "appendix", "runs")
OUTPUT_DIR <- file.path(paths$outputs, "appendix", "figures", "alluvial")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Repo mode keeps appendix figures under the analytical outputs tree only.
record_figure_asset <- function(source, exported_name) {
  invisible(source)
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_ID <- Sys.getenv("RDF_RUN_ID", unset = "rdf_triplets__gpt_5_mini__supplementary__10k__v1")
TOP_N_PAIRS <- 50
MIN_YEAR <- 1990
MIN_OBJECT_YEAR_COUNT <- 5
SAVE_OUTPUTS <- TRUE
SAVE_CLEANED_TABLE <- TRUE
DROP_WEAK_IDENTITY_ROWS <- TRUE
DOWNGRADE_AMBIGUOUS_CONSTITUTES <- TRUE
PREFER_ONTOLOGY_CLEANED <- TRUE

TRIPLET_PATH_OVERRIDE <- Sys.getenv("RDF_TRIPLET_PATH", unset = "")
default_cleaned_path <- file.path(RUNS_DIR, RUN_ID, "triplet_flat_cleaned.parquet")
default_raw_path <- file.path(RUNS_DIR, RUN_ID, "triplet_flat.parquet")
TRIPLET_PATH <- if (nzchar(TRIPLET_PATH_OVERRIDE)) {
  TRIPLET_PATH_OVERRIDE
} else if (isTRUE(PREFER_ONTOLOGY_CLEANED) && file.exists(default_cleaned_path)) {
  default_cleaned_path
} else {
  default_raw_path
}

if (!file.exists(TRIPLET_PATH)) {
  stop(
    paste0(
      "Triplet file not found: ", TRIPLET_PATH, "\n",
      "Wait for the run to finish, or point RDF_RUN_ID to a completed run."
    ),
    call. = FALSE
  )
}

# ---------------------------------------------------------------------------
# Load and inspect
# ---------------------------------------------------------------------------

read_triplet_table <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext == "parquet") {
    return(read_parquet(path))
  }
  if (ext == "csv") {
    return(read.csv(path, stringsAsFactors = FALSE, check.names = FALSE))
  }
  stop(
    paste0(
      "Unsupported triplet file extension: .", ext, "\n",
      "Use a .parquet or .csv triplet table."
    ),
    call. = FALSE
  )
}

triplets_raw <- read_triplet_table(TRIPLET_PATH)

required_cols <- c("bibcode", "year", "title", "subject", "predicate", "object", "claim_type", "evidence_text")
missing_cols <- setdiff(required_cols, names(triplets_raw))
if (length(missing_cols) > 0) {
  stop(
    sprintf(
      "Triplet file is missing required columns: %s",
      paste(missing_cols, collapse = ", ")
    ),
    call. = FALSE
  )
}

print(names(triplets_raw))
print(head(triplets_raw, 10))
print(count(triplets_raw, predicate, sort = TRUE))
print(summary(triplets_raw$year))

# ---------------------------------------------------------------------------
# Normalize and post-process labels
# ---------------------------------------------------------------------------

if ("cleaning_pipeline" %in% names(triplets_raw)) {
  triplets <- triplets_raw |>
    filter(
      !is.na(year),
      year >= MIN_YEAR,
      !is.na(subject), subject != "",
      !is.na(object), object != "",
      !is.na(predicate), predicate != "",
      !(subject == "dark matter" & predicate == "is" & object == "dark matter")
    )

  message("Using ontology-cleaned triplets from: ", TRIPLET_PATH)
  print(count(triplets, predicate, sort = TRUE))
  print(count(triplets, subject, sort = TRUE) |> head(20))
  print(count(triplets, object, sort = TRUE) |> head(20))
} else {
  message(
    "Using legacy local cleanup because no ontology-cleaned file was found. ",
    "Run clean_triplets_with_ontology.py first for the canonical pipeline."
  )

normalize_triplet_term <- function(x) {
  x <- str_squish(str_to_lower(as.character(x)))
  x <- str_replace_all(x, fixed("\u03bd"), "nu")
  x <- str_replace_all(x, fixed("γ"), "gamma")

  dplyr::case_when(
    x == "gravitino dark matter dark matter" ~ "gravitino",
    x == "self-interacting dark matter" ~ "SIDM",
    x == "primordial intermediate mas black hole" ~ "PBH",
    x %in% c("primordial black hole", "primordial black holes", "pbh", "pbhs") ~ "PBH",
    x %in% c("weakly interacting massive particle", "weakly interacting massive particles", "weakly interacting massive particle (wimp)", "weakly interacting massive particles (wimps)", "wimp", "wimps") ~ "WIMP",
    x %in% c("quantum chromodynamic axion", "qcd axion") ~ "QCD axion",
    x == "neutralino dark matter" ~ "neutralino",
    x == "massive compact halo object" ~ "MACHO",
    x %in% c("axion", "axions") ~ "axion",
    x %in% c("axion-like particle", "axion-like particles", "alp", "alps") ~ "ALP",
    x == "cold dark matter" ~ "CDM",
    x %in% c("sterile neutrino", "sterile neutrinos", "kev sterile neutrinos", "kev sterile neutrino") ~ "sterile neutrino",
    x %in% c("hidden photon", "hidden photons") ~ "hidden photon",
    x %in% c("dark photon", "dark photons") ~ "dark photon",
    x == "fermionic" ~ "fermionic dark matter",
    x == "bosonic" ~ "bosonic dark matter",
    x %in% c("lightest supersymmetric particle", "lightest supersymmetric particle (lsp)", "lsp") ~ "LSP",
    x %in% c("lightest neutralino") ~ "neutralino",
    x %in% c("self interacting dark matter", "self-interacting dark matter") ~ "self-interacting dark matter",
    x %in% c("warm dark matter") ~ "warm dark matter",
    x %in% c("fuzzy dark matter") ~ "fuzzy dark matter",
    x %in% c("fermion", "fermions") ~ "fermion",
    x %in% c("scalar particle", "scalar particles") ~ "scalar particle",
    x == "Feebly Interacting Massive Particle (FIMP)" ~ "FIMP",
    TRUE ~ x
  )
}

has_candidate_cue <- function(text) {
  t <- str_to_lower(str_squish(as.character(text)))
  str_detect(
    t,
    "candidate|viable|search for|as dark matter|for dark matter|could be dark matter|may be dark matter|might be dark matter"
  )
}

is_weak_identity_object <- function(x) {
  x <- str_to_lower(str_squish(as.character(x)))
  x %in% c(
    "cold",
    "non-baryonic",
    "baryonic",
    "fermionic",
    "self-interacting",
    "collisionless",
    "particles",
    "particle dark matter",
    "thermal relic",
    "non-luminous",
    "ultralight bosonic"
  )
}

triplets <- triplets_raw |>
  mutate(
    year = suppressWarnings(as.integer(year)),
    predicate_original = predicate,
    claim_type_original = claim_type,
    subject_original = subject,
    object_original = object,
    subject = normalize_triplet_term(subject),
    object = normalize_triplet_term(object)
  ) |>
  mutate(
    predicate = case_when(
      DOWNGRADE_AMBIGUOUS_CONSTITUTES &
        predicate == "constitutes" &
        object == "dark matter" &
        (has_candidate_cue(evidence_text) | has_candidate_cue(title)) ~ "candidate_for",
      TRUE ~ predicate
    ),
    claim_type = case_when(
      DOWNGRADE_AMBIGUOUS_CONSTITUTES &
        predicate_original == "constitutes" &
        predicate == "candidate_for" ~ "candidate",
      TRUE ~ claim_type
    )
  ) |>
  filter(
    !is.na(year),
    year >= MIN_YEAR,
    !is.na(subject), subject != "",
    !is.na(object), object != "",
    !is.na(predicate), predicate != "",
    !(subject == "dark matter" & predicate == "is" & object == "dark matter")
  )

if (isTRUE(DROP_WEAK_IDENTITY_ROWS)) {
  weak_identity_n <- triplets |>
    filter(subject == "dark matter", predicate == "is", is_weak_identity_object(object)) |>
    nrow()

  triplets <- triplets |>
    filter(!(subject == "dark matter" & predicate == "is" & is_weak_identity_object(object)))

  message("Dropped weak identity rows: ", weak_identity_n)
}

if (isTRUE(SAVE_CLEANED_TABLE)) {
  cleaned_base <- file.path(OUTPUT_DIR, paste0(RUN_ID, "__triplets_cleaned"))
  write.csv(triplets, paste0(cleaned_base, ".csv"), row.names = FALSE)
  write_parquet(triplets, paste0(cleaned_base, ".parquet"))
  message("Saved cleaned triplets to: ", cleaned_base, ".[csv|parquet]")
}

print("Top normalization changes")
print(
  triplets |>
    filter(
      subject != subject_original |
        object != object_original |
        predicate != predicate_original
    ) |>
    count(subject_original, subject, object_original, object, predicate_original, predicate, sort = TRUE) |>
    head(50)
  )

print(count(triplets, predicate, sort = TRUE))
print(count(triplets, claim_type, sort = TRUE))
print(count(triplets, subject, sort = TRUE) |> head(50), n = 50)
print(count(triplets, object, sort = TRUE) |> head(50), n = 50)
}

# ---------------------------------------------------------------------------
# Plot 1: top subject-object links
# ---------------------------------------------------------------------------

pair_counts_top <- triplets |>
  count(subject, object, sort = TRUE, name = "Count") |>
  slice_max(Count, n = 12, with_ties = FALSE)

print(pair_counts_top)

p_pairs <- ggplot(
  pair_counts_top,
  aes(axis1 = subject, axis2 = object, y = Count)
) +
  geom_alluvium(aes(fill = Count), size = 0.25, color = "white",width = 1 / 12, alpha = 0.75, curve_type = "sigmoid") +
  geom_stratum(width = 1.5 / 12, fill = "grey90", size = 0.25, color = "white", alpha = 1) +
  geom_text(
    stat = "stratum",
    aes(label = after_stat(stratum)),
    size = 2.5,
    hjust = 0.5,
    family = "Helvetica"
  ) +
  scale_fill_carto_c(
    name = "Count",
    type = "diverging",
    palette = "BluGrn",
    direction = 1
  ) +
  scale_x_discrete(limits = c("Subject", "Object"), expand = c(0.05, 0.05)) +
  scale_y_continuous(labels = scales::comma) +
  theme_void(base_family = "Helvetica", base_size = 9) +
  labs(
    title = "Top subject-object relationships in the triplet extraction"
  ) +
  theme(
    legend.position = c(0.5, 0.01),
    plot.margin = margin(0.5, 0.5, 1.5, 0.5, "cm"),
    legend.title.position = "left",
    legend.title = element_text(size = 9, family = "Helvetica"),
    legend.text = element_text(size = 8, family = "Helvetica"),
    plot.title = element_text(
      size = 11,
      hjust = 0.5,
      family = "Helvetica",
      vjust = -3.0,
      margin = margin(0, 0, 0, 0)
    )
  ) +
  guides(fill = guide_colourbar(
    direction = "horizontal",
    title.hjust = 0.5,
    title.vjust = 1.0,
    barwidth = unit(4.5, "cm"),
    barheight = unit(0.3, "cm")
  ))

print(p_pairs)

if (isTRUE(SAVE_OUTPUTS)) {
  pairs_output_path <- file.path(OUTPUT_DIR, paste0(RUN_ID, "__top_subject_object_pairs.pdf"))
  ggsave(
    filename = pairs_output_path,
    plot = p_pairs,
    width = 8.0,
    height = 4.8,
    units = "in"
  )
  record_figure_asset(pairs_output_path, "rdf_top_subject_object_pairs.pdf")
}

# ---------------------------------------------------------------------------
# Plot 2: what dark matter is linked to over time
# ---------------------------------------------------------------------------

flow1 <- triplets |>
  filter(object == "dark matter") |>
  transmute(year, object_new = subject)

flow2 <- triplets |>
  filter(subject == "dark matter") |>
  transmute(year, object_new = object)

flow_combined <- bind_rows(flow1, flow2)

object_year_counts <- flow_combined |>
  count(year, object_new, sort = TRUE) |>
  filter(n > 5)

plot_title_over_time <- "How dark matter is linked to candidate or identity terms over time"

if (nrow(object_year_counts) == 0) {
  stop(
    paste(
      "No dark-matter-linked yearly counts survived the current threshold.",
      "Lower MIN_OBJECT_YEAR_COUNT or rerun on a larger triplet set."
    ),
    call. = FALSE
  )
}

object_totals <- object_year_counts |>
  count(object_new, wt = n, sort = TRUE)

object_order <- object_totals |>
  pull(object_new)

year_order <- object_year_counts |>
  pull(year) |>
  unique() |>
  sort()

object_year_counts <- object_year_counts |>
  filter(year != 2026) |>
  mutate(
    object_new = factor(object_new, levels = object_order),
    year = factor(year, levels = year_order)
  )

print(head(object_year_counts, 20))

n_objects <- nlevels(object_year_counts$object_new)

p_dark_matter_over_time <- ggplot(
  object_year_counts,
  aes(x = year, y = n, fill = object_new)
) +
  geom_col(position = position_stack(reverse = TRUE)) +
  scale_fill_paletteer_d("ggthemes::Green_Orange_Teal") +
  labs(
    title = plot_title_over_time,
    x = NULL,
    y = "Count",
    fill = "Linked term"
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.05))
    ) +
  theme_minimal(base_size = 9) +
  theme(
    panel.grid.major.x = element_blank(),
    legend.position = "bottom",
    legend.key.size = unit(0.8, "lines"),
    legend.key.spacing.y = unit(0.4, "lines"),
    legend.title = element_text(size = 0),
    legend.text = element_text(size = 9),
    plot.title = element_text(size = 11, hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  guides(fill = guide_legend(ncol = 9, reverse = FALSE))

print(p_dark_matter_over_time)

if (isTRUE(SAVE_OUTPUTS)) {
  over_time_output_path <- file.path(OUTPUT_DIR, paste0(RUN_ID, "__dark_matter_over_time.pdf"))
  ggsave(
    filename = over_time_output_path,
    plot = p_dark_matter_over_time,
    width = 9.0,
    height = 5.4,
    units = "in"
  )
  record_figure_asset(over_time_output_path, "rdf_dark_matter_over_time.pdf")
}

message("\nPrepared alluvial analysis for run: ", RUN_ID)
message("Triplet source: ", TRIPLET_PATH)
message("Output directory: ", OUTPUT_DIR)
if (isTRUE(SAVE_OUTPUTS)) {
  message("Analytical figure outputs: ", OUTPUT_DIR)
}
if (!isTRUE(SAVE_OUTPUTS)) {
  message("Plots were printed but not saved. Set SAVE_OUTPUTS <- TRUE to write PDFs.")
}
