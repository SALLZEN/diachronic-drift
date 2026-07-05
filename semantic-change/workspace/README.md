

# Semantic Change Workspace

This workspace is the canonical computational replication surface for the
`semantic-change` project.

## Canonical structure

- `code/notebooks/` for the main notebook spine
- `code/scripts/` for Python and R entrypoints
- `code/appendix/` for the supplementary RDF / triplet branch
- `config/` for dependency contracts and the workspace root marker
- `data/` for standardized project-local data products
- `outputs/` for retained analytical outputs
- `docs/` for runbooks and structure maps
- `local/` for durable project-specific assets that are not shared
  cross-project

## Accepted notebook spine

- `0.0.0-master-corpus-build.ipynb`
- `0.1.0-corpus-splitting.ipynb`
- `1.0.0-embeddings.ipynb`
- `1.1.0-dimension-reduction.ipynb`
- `2.0.0-seed-classification-gpt.ipynb`
  In repo mode, the intended default is cached `full_run` + `replay` so the downstream notebook sequence can be reproduced without new OpenAI calls; `run_1` / `run_2` remain available for rerunning the 200-item reliability stage.
- `2.1.0-train-classification-model.ipynb`
- `2.2.0-run-classification-model.ipynb`
- `3.0.0-empirical-testing.ipynb`
- `3.1.0-significance-testing.ipynb`
- `4.0.0-collostructional-analysis.ipynb`

## Accepted scripts and outputs

- `code/scripts/setup_scibert_model.py`
- `outputs/analytical-results/`
- `outputs/dimension-reduction/`
- `outputs/appendix/`

## Notes

- all executable surfaces resolve through `config/workspace.json` and
  `../../shared-assets/code/workspace_rooting/`
- all Python notebooks in this bundle assume the local
  `semantic-change-repo` kernel backed by `../../.venv/bin/python`
- when this workspace is used through a repo export, run
  `code/scripts/setup_scibert_model.py` once before
  `1.0.0-embeddings.ipynb` so the local SciBERT directory exists at
  `local/models/scibert_scivocab_uncased`; the script unpacks the
  bundled base-model archive at
  `../shared-assets/models/scibert_uncased/scibert_bundle.zip`
- notebook rendering/export helpers are not bundled in this repo export;
  treat the notebooks themselves as the canonical executable surfaces
- no separate publication-facing surface is retained in this
  code-replication bundle

The kernel install step is occasional setup only, not something to run
before each notebook session.
