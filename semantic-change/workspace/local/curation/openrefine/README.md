

# OpenRefine Sidecar

This directory preserves a manual OpenRefine pass over experimental RDF
triplets.

Contents:

- `triplet-flat-OR.csv` Manually normalized export produced in
  OpenRefine.
- `history.json` Recorded OpenRefine operations.
- `triplet-flat-csv.openrefine.tar.gz` OpenRefine project archive.

Role in the pipeline:

- This directory is **archival and comparative**, not the canonical live
  dependency.
- The canonical reproducible path is now:
  1.  extract raw triplets
  2.  run `APDX_2.0.1-clean-triplets.ipynb`
  3.  inspect / visualize the cleaned output
- The OpenRefine material remains useful as:
  - a manual benchmark
  - a source of additional normalization ideas
  - an audit trail for human intervention

If you want to compare the ontology-cleaned output against the manual
pass, point `APDX_2.0.1-clean-triplets.ipynb` or
`APDX_3.0.0-build-triplet-assets.R` at `triplet-flat-OR.csv`.
