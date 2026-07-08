"""Backend helpers for notebook-first RDF triplet cleaning.

The same cleaning stage is implemented in:

`APDX_2.0.1-clean-triplets.ipynb`

This module provides the reusable logic behind that notebook and the optional
CLI entrypoint.

It bridges:

1. raw triplet outputs from the experimental RDF notebook
2. the shared term-normalization code in `shared-assets/code`
3. downstream exploratory analysis such as `APDX_3.0.0-build-triplet-assets.R`

It is intentionally conservative. The script prefers:
- canonicalizing recurring candidate/entity labels
- preserving raw text alongside cleaned values
- filtering or downgrading only the most obviously weak ontology claims
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


SHARED_CODE_DIR = Path(__file__).resolve().parents[4] / "shared-assets" / "code"
if str(SHARED_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE_DIR))

from workspace_rooting.workspace_paths import canonical_workspace_paths

PATHS = canonical_workspace_paths(Path(__file__))
SCRIPT_DIR = PATHS["code"] / "appendix"
RUNS_DIR = PATHS["outputs"] / "appendix" / "runs"
OPENREFINE_DIR = PATHS["local"] / "curation" / "openrefine"

from dm_term_normalization.candidate_ontology import (  # noqa: E402
    DM_TAGS_INFO,
    SPECIES_LABEL_MAP,
    candidate_species_labels,
    extract_dm_tags_with_spans,
)
from dm_term_normalization.normalization import normalize_segment  # noqa: E402


REQUIRED_COLUMNS = [
    "bibcode",
    "year",
    "title",
    "subject",
    "predicate",
    "object",
    "claim_type",
    "evidence_text",
]


DISPLAY_LABEL_OVERRIDES = {
    "Generic WIMP": "WIMP",
    "Axion + ALP": "axion/ALP",
    "QCD Axion": "QCD axion",
    "Sterile nu": "sterile neutrino",
    "Macroscopic DM": "macroscopic dark matter",
    "Asymmetric DM": "asymmetric dark matter",
    "Atomic DM": "atomic dark matter",
    "Mirror DM": "mirror dark matter",
    "Twin Higgs": "twin Higgs",
    "Dark Photon": "dark photon",
    "Vector DM": "vector dark matter",
    "LKP Kaluza-Klein": "Kaluza-Klein",
    "Quark Nugget / AQN": "quark nugget / AQN",
    "Planckian DM": "planckian dark matter",
    "ULA / Fuzzy": "ULA / fuzzy dark matter",
    "SFDM": "scalar dark matter",
    "Warm (WDM)": "warm dark matter",
    "Self-Interacting (SIDM)": "SIDM",
    "Fuzzy / waveDM": "ULA / fuzzy dark matter",
    "Superfluid": "superfluid dark matter",
    "Feebly Interacting Massive Particle (FIMP)": "FIMP",
    "Strongly Interacting Massive Particle (SIMP)": "SIMP",
    "Inelastic (iDM)": "iDM",
    "Light (sub-GeV/MeV-scale)": "light DM",
    "Mirror DM": "mirror DM",
    "Generic Composite": "composite dark matter",
}


CANONICAL_ALIASES = {
    "χ": "chi",
    "chi": "chi",
    "dark matter": "dark matter",
    "cold dark matter": "CDM",
    "cold dark matter (cdm)": "CDM",
    "cdm": "CDM",
    "lambda cold dark matter": "Lambda CDM",
    "lambda-cold dark matter": "Lambda CDM",
    "lambda cdm": "Lambda CDM",
    "lcdm": "Lambda CDM",
    "warm dark matter": "warm dark matter",
    "warm dark matter (wdm)": "warm dark matter",
    "wdm": "warm dark matter",
    "fuzzy dark matter": "fuzzy dark matter",
    "fuzzy dark matter (fdm)": "fuzzy dark matter",
    "fdm": "fuzzy dark matter",
    "scalar-field dark matter": "scalar dark matter",
    "scalar field dark matter": "scalar dark matter",
    "self-interacting dark matter": "SIDM",
    "self interacting dark matter": "SIDM",
    "self-interacting dark matter (sidm)": "SIDM",
    "sidm": "SIDM",
    "mirror dark matter": "mirror DM",
    "mirror matter": "mirror DM",
    "primordial black hole": "PBH",
    "primordial black holes": "PBH",
    "primordial black holes (pbhs)": "PBH",
    "pbh": "PBH",
    "pbhs": "PBH",
    "weakly interacting massive particle": "WIMP",
    "weakly interacting massive particles": "WIMP",
    "weakly interacting massive particle (wimp)": "WIMP",
    "weakly interacting massive particles (wimp)": "WIMP",
    "weakly interacting massive particles (wimps)": "WIMP",
    "wimp": "WIMP",
    "wimps": "WIMP",
    "qcd axion": "QCD axion",
    "axion": "axion",
    "axions": "axion",
    "axion-like particle": "ALP",
    "axion-like particles": "ALP",
    "axionlike particle": "ALP",
    "axionlike particles": "ALP",
    "alp": "ALP",
    "alps": "ALP",
    "neutralinos": "neutralino",
    "massive neutrinos": "neutrino",
    "neutrinos": "neutrino",
    "right-handed neutrinos": "right-handed neutrino",
    "lightest right-handed neutrino": "right-handed neutrino",
    "lightest right-handed neutrinos": "right-handed neutrino",
    "sterile-like neutrino": "sterile neutrino",
    "sterile-like neutrinos": "sterile neutrino",
    "sterile neutrinos": "sterile neutrino",
    "kev sterile neutrinos": "sterile neutrino",
    "kev sterile neutrino": "sterile neutrino",
    "hidden photons": "hidden photon",
    "dark photons": "dark photon",
    "majorana fermions": "majorana fermion",
    "dirac fermions": "dirac fermion",
    "scalar particles": "scalar particle",
    "scalar singlet": "singlet scalar",
    "fermions": "fermion",
    "particles": "particle",
    "feebly interacting massive particle": "FIMP",
    "feebly interacting massive particle (fimp)": "FIMP",
    "strongly interacting massive particle": "SIMP",
    "strongly interacting massive particle (simp)": "SIMP",
    "inelastic dark matter": "iDM",
    "inelastic (idm)": "iDM",
    "light (sub-gev/mev-scale)": "light DM",
    "light dark matter": "light DM",
    "supersymmetry": "SUSY",
    "supersymmetric particles": "SUSY particles",
    "fermionic dark matter candidate": "fermionic dark matter",
}


WEAK_IDENTITY_OBJECTS = {
    "cold",
    "non-baryonic",
    "nonbaryonic",
    "baryonic",
    "fermionic",
    "bosonic",
    "self-interacting",
    "collisionless",
    "particle",
    "particles",
    "particle dark matter",
    "thermal relic",
    "non-luminous",
    "ultralight bosonic",
    "multiple components",
    "two components",
    "a pressureless perfect fluid",
    "dust",
    "invisible",
    "dissipationless",
    "dissipative",
    "fermion",
    "scalar",
    "scalar particle",
    "scalar field",
    "neutrino",
    "majorana fermion",
    "dirac fermion",
    "complex scalar",
    "fermionic particles",
    "supersymmetry",
    "susy particles",
    "cold and hot components",
    "cold dark matter and hot dark matter",
    "cold dark matter fluid",
}


CANDIDATE_CUE_RE = re.compile(
    r"candidate|viable|search for|as dark matter|for dark matter|"
    r"could be dark matter|may be dark matter|might be dark matter",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rdf_triplets__gpt_5_mini__supplementary__10k__v1")
    parser.add_argument("--input", help="Optional explicit input triplet file (.parquet or .csv)")
    parser.add_argument("--output-base", help="Optional explicit output basename without extension")
    parser.add_argument("--min-year", type=int, default=1990)
    parser.add_argument("--source-label", default="shared-ontology-cleaner")
    parser.add_argument(
        "--prefer-openrefine",
        action="store_true",
        help="If set and no --input is provided, prefer openrefine/triplet-flat-OR.csv when present.",
    )
    return parser.parse_args()


def default_input_path(run_id: str, prefer_openrefine: bool) -> Path:
    if prefer_openrefine and (OPENREFINE_DIR / "triplet-flat-OR.csv").exists():
        return OPENREFINE_DIR / "triplet-flat-OR.csv"
    return RUNS_DIR / run_id / "triplet_flat.parquet"


def default_output_base(input_path: Path, run_id: str) -> Path:
    if input_path.parent == OPENREFINE_DIR:
        stem = input_path.stem
        return input_path.with_name(f"{stem}.cleaned")
    return RUNS_DIR / run_id / "triplet_flat_cleaned"


def read_triplets(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def validate_triplets(triplets_raw: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in triplets_raw.columns]
    if missing:
        raise ValueError(f"Triplet input missing required columns: {', '.join(missing)}")


def has_candidate_cue(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(CANDIDATE_CUE_RE.search(text))


def basic_term_normalize(value: str) -> str:
    if not isinstance(value, str):
        return ""
    value = normalize_segment(value.lower(), mask_math=True)
    value = re.sub(r"\s+", " ", value).strip(" .,;:()[]{}")
    value = value.replace("nu", "nu").strip()
    return CANONICAL_ALIASES.get(value, value)


def ontology_tags(value: str) -> list[str]:
    if not value:
        return []
    tags, _ = extract_dm_tags_with_spans(value)
    return tags


def compact_label_from_tags(tags: list[str]) -> str | None:
    if not tags:
        return None
    labels = candidate_species_labels(tags)
    if labels:
        compact = " + ".join(DISPLAY_LABEL_OVERRIDES.get(label, label) for label in labels)
        return compact

    fallback_labels = []
    for tag in tags:
        full = DM_TAGS_INFO.get(tag, {}).get("full")
        if full:
            fallback_labels.append(full)
    if fallback_labels:
        return " + ".join(fallback_labels)
    return None


def canonicalize_term(value: str) -> tuple[str, list[str], str | None]:
    normalized = basic_term_normalize(value)
    if normalized in CANONICAL_ALIASES:
        normalized = CANONICAL_ALIASES[normalized]

    tags = ontology_tags(normalized)
    compact = compact_label_from_tags(tags)

    if compact:
        return compact, tags, compact
    return normalized, tags, None


def is_weak_identity_object(value: str) -> bool:
    return basic_term_normalize(value) in WEAK_IDENTITY_OBJECTS


def summarize_openrefine_history(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        ops = json.loads(path.read_text())
    except Exception:
        return {"path": str(path), "readable": False}

    descriptions = []
    if isinstance(ops, list):
        descriptions = [op.get("description", "") for op in ops if isinstance(op, dict)]
    return {
        "path": str(path),
        "readable": True,
        "operation_count": len(descriptions),
        "descriptions": descriptions,
    }


def resolve_paths(
    *,
    run_id: str,
    input_path: str | Path | None = None,
    output_base: str | Path | None = None,
    prefer_openrefine: bool = False,
) -> tuple[Path, Path]:
    resolved_input = Path(input_path).expanduser().resolve() if input_path else default_input_path(run_id, prefer_openrefine)
    if not resolved_input.exists():
        raise FileNotFoundError(f"Triplet input not found: {resolved_input}")

    resolved_output = (
        Path(output_base).expanduser().resolve()
        if output_base
        else default_output_base(resolved_input, run_id)
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    return resolved_input, resolved_output


def clean_triplets_frame(
    triplets_raw: pd.DataFrame,
    *,
    run_id: str,
    min_year: int = 1990,
    source_label: str = "shared-ontology-cleaner",
    openrefine_history_path: Path | None = None,
    input_path: Path | None = None,
    output_base: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_triplets(triplets_raw)

    triplets = triplets_raw.copy()
    triplets["subject_raw"] = triplets["subject"].astype("string")
    triplets["object_raw"] = triplets["object"].astype("string")
    triplets["predicate_raw"] = triplets["predicate"].astype("string")
    triplets["claim_type_raw"] = triplets["claim_type"].astype("string")
    triplets["year"] = pd.to_numeric(triplets["year"], errors="coerce").astype("Int64")

    subject_terms = triplets["subject_raw"].fillna("").map(canonicalize_term)
    object_terms = triplets["object_raw"].fillna("").map(canonicalize_term)

    triplets["subject"] = subject_terms.map(lambda item: item[0])
    triplets["subject_tags"] = subject_terms.map(lambda item: "; ".join(item[1]))
    triplets["subject_label"] = subject_terms.map(lambda item: item[2] or "")
    triplets["object"] = object_terms.map(lambda item: item[0])
    triplets["object_tags"] = object_terms.map(lambda item: "; ".join(item[1]))
    triplets["object_label"] = object_terms.map(lambda item: item[2] or "")

    triplets["predicate"] = triplets["predicate"].astype("string").str.strip()
    triplets["claim_type"] = triplets["claim_type"].astype("string").str.strip()

    constitutes_mask = (
        (triplets["predicate"] == "constitutes")
        & (triplets["object"] == "dark matter")
        & (
            triplets["evidence_text"].fillna("").map(has_candidate_cue)
            | triplets["title"].fillna("").map(has_candidate_cue)
            | triplets["subject_tags"].fillna("").str.len().gt(0)
        )
    )
    triplets.loc[constitutes_mask, "predicate"] = "candidate_for"
    triplets.loc[constitutes_mask, "claim_type"] = "candidate"

    weak_identity_mask = (
        (triplets["subject"] == "dark matter")
        & (triplets["predicate"] == "is")
        & triplets["object"].fillna("").map(is_weak_identity_object)
    )

    triplets = triplets.loc[
        triplets["year"].notna()
        & (triplets["year"] >= min_year)
        & triplets["subject"].fillna("").ne("")
        & triplets["object"].fillna("").ne("")
        & triplets["predicate"].fillna("").ne("")
        & ~(triplets["subject"].eq("dark matter") & triplets["predicate"].eq("is") & triplets["object"].eq("dark matter"))
        & ~weak_identity_mask
    ].copy()

    triplets["cleaning_pipeline"] = source_label

    summary = {
        "input_path": str(input_path) if input_path is not None else None,
        "output_base": str(output_base) if output_base is not None else None,
        "run_id": run_id,
        "row_count_raw": int(len(triplets_raw)),
        "row_count_cleaned": int(len(triplets)),
        "downgraded_constitutes_to_candidate_for": int(constitutes_mask.sum()),
        "dropped_weak_identity_rows": int(weak_identity_mask.sum()),
        "predicate_counts": triplets["predicate"].value_counts(dropna=False).to_dict(),
        "top_subjects": triplets["subject"].value_counts(dropna=False).head(20).to_dict(),
        "top_objects": triplets["object"].value_counts(dropna=False).head(20).to_dict(),
        "openrefine_history": summarize_openrefine_history(openrefine_history_path or (OPENREFINE_DIR / "history.json")),
    }

    return triplets, summary


def write_cleaned_outputs(
    triplets: pd.DataFrame,
    summary: dict[str, object],
    *,
    output_base: Path,
) -> None:
    triplets.to_csv(output_base.with_suffix(".csv"), index=False)
    triplets.to_parquet(output_base.with_suffix(".parquet"), index=False)
    output_base.with_name(output_base.name + "_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    input_path, output_base = resolve_paths(
        run_id=args.run_id,
        input_path=args.input,
        output_base=args.output_base,
        prefer_openrefine=args.prefer_openrefine,
    )
    triplets_raw = read_triplets(input_path)
    triplets, summary = clean_triplets_frame(
        triplets_raw,
        run_id=args.run_id,
        min_year=args.min_year,
        source_label=args.source_label,
        openrefine_history_path=OPENREFINE_DIR / "history.json",
        input_path=input_path,
        output_base=output_base,
    )
    write_cleaned_outputs(triplets, summary, output_base=output_base)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
