

# Appendix RDF Extraction

This directory is the accepted supplementary RDF / triplet branch for
the `semantic-change` project.

The goal is to keep the retained RDF branch reproducible, resumable, and
clearly subordinate to the main notebook spine:

- input: `../../data/parquet/ADS_MAIN_FRAME_2025.parquet`
- API pattern: OpenAI Responses API
- default model: `gpt-5-mini`
- output style: structured JSON triplets, cached as JSONL and flattened
  to tabular files

## Files

- `APDX_1.0.0-rdf-triplet-extraction-api.ipynb` The main appendix
  notebook.
- `APDX_2.0.1-clean-triplets.ipynb` Canonical notebook for
  post-extraction triplet cleaning with the shared term-normalization
  layer.
- `rdf_triplet_extraction.py` Small helper module for prompting, schema,
  caching, and output flattening.
- `clean_triplets_with_ontology.py` Backend/helper module used by the
  cleaning notebook and optional CLI runs.
- `../../local/curation/openrefine/` Preserved manual cleanup sidecar.
  Useful for comparison and audit, but no longer the default live
  dependency.

## Why `gpt-5-mini`

The extraction task is tightly specified, repetitive, and
cost-sensitive. `gpt-5-mini` is a strong default for this kind of
well-defined structured task, while still supporting Structured Outputs.
If precision checks suggest the need for a stronger model, the same
notebook can be rerun with `MODEL = "gpt-5"`.

## Inputs

- `../data/parquet/ADS_MAIN_FRAME_2025.parquet` The project-local ADS
  frame built from the shared project backbone.

Expected columns:

- `bibcode`
- `year`
- `title`
- `abstract`
- `doctype`

## Recommended supplementary run

Once the adjudication sample looks acceptable, use exactly this launch
profile for the retained composite run:

- `MODEL = "gpt-5-mini"`
- `SAMPLE_N = None`
- `RUN_ID = "rdf_triplets__gpt_5_mini__supplementary__10k__v1"`
- `FORCE_REPROCESS = False`
- `ABSTRACT_MODE = "raw_normalized"`
- `SUPPLEMENTARY_FILTER = True`
- `SUPPLEMENTARY_YEAR_MIN = 1990`
- `SUPPLEMENTARY_CUE_REGEX = r"candidate|viable|as dark matter|dark matter candidate|constitut|compos|consist(?:s|ing)? of|made of|wimp|axion|neutralino|sterile neutrino|hidden photon|primordial black hole|self[ -]interacting dark matter|warm dark matter|cold dark matter|fuzzy dark matter|fermionic dark matter|bosonic dark matter|scalar dark matter"`
- `COMPOSITE_SEED_RUN_ID = "rdf_triplets__gpt_5_mini__supplementary__v2"`
- `COMPOSITE_SEED_N = 800` from cached `1990–2000` rows
- `COMPOSITE_FRESH_TARGETS = {"2001_2009": 1700, "2010_2019": 3500, "2020_2029": 4000}`
- `MAX_WORKERS = 4`

This keeps the run cheap enough to be practical, conservative enough to
be methodologically defensible, and cleanly resumable if interrupted. It
also narrows the corpus to the modern ontology-interest literature,
while preserving a random subset of the early `1990–2000` rows you
already paid for instead of rerunning them. The notebook writes a
combined retained run whose flattened outputs include both the cached
800-row seed sample and the fresh 9,200-row stratified sample.

Suggested sequence:

1.  Set the supplementary preset in the notebook config cell.
2.  Run the notebook top to bottom.
3.  Confirm the cached seed sample and fresh period targets before
    starting the long extraction cell.
4.  Run the fresh 9,200-row extraction.
5.  Run `APDX_2.0.1-clean-triplets.ipynb` on the finished
    `triplet_flat.parquet`.
6.  Inspect predicate counts and a small random spot-check at the end.
7.  Treat the outputs as supplementary unless a later manual review
    suggests a stronger role.

## Outputs

The notebook writes into `../../outputs/appendix/runs/<run_id>/`:

- `seed_triplet_requests.jsonl` The retained 800-row seed sample drawn
  from the cached `1990–2000` run.
- `triplet_requests.jsonl` One JSON object per freshly processed
  abstract, including model metadata and extraction result.
- `triplet_flat.parquet` Flattened triplets table for the full combined
  10K run.
- `triplet_flat.csv` CSV export of the flattened full-run table.
- `triplet_flat_cleaned.parquet` Ontology-cleaned canonical table for
  downstream appendix analysis.
- `triplet_flat_cleaned.csv` CSV export of the ontology-cleaned table.
- `run_manifest.json` Run metadata, counts, model choice, prompt hash,
  and composite-seed details.
- `seed_selection.json` Metadata describing the cached seed sample and
  fresh target quotas.

## Progress monitoring

This standalone repo export does not retain a separate monitor utility
as a canonical appendix surface. Progress should instead be read
directly from the JSONL request log and run manifest inside
`../../outputs/appendix/runs/<run_id>/`.

## Post-extraction cleaning

Once a run has written `triplet_flat.parquet`, clean it in:

- `APDX_2.0.1-clean-triplets.ipynb`

That notebook writes into the same retained run directory:

- `triplet_flat_cleaned.parquet`
- `triplet_flat_cleaned.csv`
- `triplet_flat_cleaned_summary.json`

The cleaning notebook is the main place where the shared resource is
used for the triplets. It imports the centralized normalization and
candidate-label logic from `shared-assets/code/dm_term_normalization/`,
canonicalizes subject/object labels, downgrades weak `constitutes`
claims when appropriate, and removes the most obviously weak identity
rows.

The Python file `clean_triplets_with_ontology.py` remains available as a
backend helper and optional CLI path, but it is no longer the intended
human-facing run surface.

`APDX_3.0.0-build-triplet-assets.R` now prefers the ontology-cleaned
table automatically when it is available.

## Notes

- The notebook is resumable: already-processed `bibcode`s are skipped
  unless a forced rerun is requested.
- API keys are read from environment variables, not embedded in notebook
  cells.
- The recommended input mode is raw abstracts with only light
  normalization of whitespace and null bytes. Do not use aggressively
  cleaned abstracts for this task. Constitutive extraction depends on
  preserving hedges, candidate language, and local syntactic cues that
  aggressive cleaning can erase or distort.
- The schema intentionally restricts predicates to a small normalized
  set: `is`, `constitutes`, `composed_of`, `candidate_for`. That makes
  downstream normalization and auditing much easier than free-form
  predicate text.
- The helper now also applies a post-response normalization pass. In
  particular, it filters:
  - invalid `constitutes` triplets whose object is not exactly
    `dark matter`
  - invalid `composed_of` triplets whose subject is not exactly
    `dark matter`
  - likely non-ontological `is` claims such as mass-budget or morphology
    statements
- When `SAMPLE_N` is used for adjudication, the notebook draws a
  period-stratified sample rather than simply taking the first `N` rows.
- Before scaling to a full paid run, use the adjudication protocol on
  the first `25` to `50` abstracts.
- The workflow is intentionally supplementary. It feeds the retained
  appendix RDF branch in this bundle, but it is not treated as a main
  inferential pillar of the replication workflow.
- The OpenRefine sidecar is kept under
  `../../local/curation/openrefine/`, not as the canonical live cleaning
  dependency.
