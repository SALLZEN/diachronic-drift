

# local

`semantic-change` requires a richer `local/` surface than the smaller
shared-workspace projects.

Canonical supplemental subdirectories: -
`models/scibert_scivocab_uncased/` - `corpora/` - `curation/openrefine/`

These are project-specific assets and manual resources that should not be
forced into the cross-project standard surfaces.

The earlier placeholders for `csv/`, `phrase_corpora/`, and
`local/models/tfidf/` were removed. The rebuilt structure will use
direct canonical surfaces rather than compatibility interfaces, and the
TF-IDF resources are treated as shared where the repo already proves
they are shared.

Current migrated local asset families: - bundled SciBERT under
`models/scibert_scivocab_uncased/` - retained corpora under `corpora/` -
retained OpenRefine curation materials under `curation/openrefine/`
