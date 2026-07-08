"""Shared helpers for the notebook-first RDF extraction workflow.

This module keeps the extraction contract stable across reruns by acting as a root source for:

- the extraction prompt
- the structured-output schema
- abstract normalization and input preparation
- the resumable JSONL and flattened output layout
- conservative post-response normalization rules
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_PROMPT = """
You extract compact RDF-style triplets from scientific abstracts about dark matter.

Your task is not to summarize the abstract. Your task is to identify explicit constitutive, identity, or candidate-style claims about what dark matter is, what could constitute it, or which entity is proposed as a candidate for it.

Work conservatively. If the abstract does not make an explicit constitutive or candidate-style claim, return an empty list.

Extraction rules:
- Every triplet must include "dark matter" as subject or object.
- Extract only claims about identity, constitution, composition, or candidacy.
- Do not extract generic phenomenology, methodology, evidence, detection strategy, parameter fitting, or observational constraints unless the sentence also makes an explicit constitutive claim.
- Do not upgrade weak language into strong language. Preserve whether a claim is identity-like, constitutive, or merely candidate-like.
- Do not treat system-level mass-budget, distribution, morphology, or location claims as ontology claims.
  For example, reject statements of the form:
  - dark matter constitutes a large fraction of the mass of a system
  - dark matter forms a belt, halo, cloud, component, or distribution
  unless the abstract explicitly presents these as identity or composition claims.
- Do not use placeholders such as "this model", "we", "the authors", "it", or "the theory". Resolve to an explicit scientific entity when possible; otherwise omit the triplet.
- Prefer short, canonical entity names over long paraphrases.
- Copy `evidence_text` directly from the abstract as a compact supporting span.

Use only the following normalized predicates:
- "is"
- "constitutes"
- "composed_of"
- "candidate_for"

Orientation rules:
- If the abstract says dark matter is X, use subject="dark matter", predicate="is", object=X.
- If the abstract says X constitutes dark matter, use subject=X, predicate="constitutes", object="dark matter".
- If the abstract says dark matter is composed of X, use subject="dark matter", predicate="composed_of", object=X.
- If the abstract says X is a candidate for dark matter, use subject=X, predicate="candidate_for", object="dark matter".

Additional conservatism rules:
- Use `constitutes` only when the object is exactly `dark matter`.
- Use `composed_of` only when the subject is exactly `dark matter`.
- If the claim is weaker than identity or constitution, prefer `candidate_for`.
"""


TRIPLET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "triplets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {
                        "type": "string",
                        "enum": ["is", "constitutes", "composed_of", "candidate_for"],
                    },
                    "object": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": ["identity", "constitution", "composition", "candidate"],
                    },
                    "evidence_text": {"type": "string"},
                },
                "required": ["subject", "predicate", "object", "claim_type", "evidence_text"],
            },
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["triplets", "notes"],
}


USEFUL_DOCTYPES = {
    "article",
    "book",
    "eprint",
    "inbook",
    "inproceedings",
    "phdthesis",
}


def prompt_hash(system_prompt: str, schema: dict[str, Any]) -> str:
    payload = json.dumps(
        {"system_prompt": system_prompt, "schema": schema},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_abstract(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.replace("\x00", " ")
    return text


def prepare_input_frame(
    df: pd.DataFrame,
    min_chars: int = 80,
    abstract_mode: str = "raw_normalized",
) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["bibcode", "abstract"]).copy()
    if "doctype" in out.columns:
        out = out[out["doctype"].fillna("").isin(USEFUL_DOCTYPES)].copy()
    out["abstract_raw"] = out["abstract"].astype(str)
    out["abstract_prompt"] = out["abstract_raw"].map(normalize_abstract)
    if abstract_mode != "raw_normalized":
        raise ValueError(f"Unsupported abstract_mode: {abstract_mode}")
    out = out[out["abstract_prompt"].str.len() >= min_chars].copy()
    out = out.sort_values(["year", "bibcode"], kind="mergesort").reset_index(drop=True)
    return out


def ensure_run_dirs(run_root: Path) -> dict[str, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    return {
        "root": run_root,
        "requests_jsonl": run_root / "triplet_requests.jsonl",
        "flat_parquet": run_root / "triplet_flat.parquet",
        "flat_csv": run_root / "triplet_flat.csv",
        "manifest_json": run_root / "run_manifest.json",
    }


def load_completed_bibcodes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            bibcode = row.get("bibcode")
            if bibcode:
                done.add(str(bibcode))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_dark_matter_term(text: str) -> bool:
    return str(text).strip().lower() == "dark matter"


def _looks_like_system_level_object(text: str) -> bool:
    t = str(text).strip().lower()
    blocked_phrases = [
        "a large fraction of the mass",
        "fraction of the mass",
        "mass of this system",
        "mass of the system",
        "part of the mass",
        "the mass of this system",
        "the mass of the system",
        "a continuous belt",
        "belt",
        "halo",
        "distribution",
        "component of the universe",
        "fraction of the universe",
    ]
    return any(phrase in t for phrase in blocked_phrases)


def _has_candidate_cue(text: str) -> bool:
    t = str(text).strip().lower()
    cues = [
        "candidate for dark matter",
        "candidate for the dark matter",
        "dark matter candidate",
        "viable dark matter candidate",
        "viable candidate",
        "as dark matter",
        "for dark matter",
        "could be dark matter",
        "may be dark matter",
        "might be dark matter",
        "best guess for such dark matter",
        "search for",
    ]
    return any(cue in t for cue in cues)


def _is_generic_identity_object(text: str) -> bool:
    t = str(text).strip().lower()
    generic_terms = {
        "fermion",
        "fermions",
        "particle",
        "particles",
        "scalar",
        "neutral scalar",
        "scalar particle",
        "neutral scalar particle",
        "boson",
        "field",
        "component",
    }
    return t in generic_terms


def normalize_triplets(triplets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    notes: list[str] = []

    for triplet in triplets:
        subject = str(triplet.get("subject") or "").strip()
        predicate = str(triplet.get("predicate") or "").strip()
        obj = str(triplet.get("object") or "").strip()
        claim_type = str(triplet.get("claim_type") or "").strip()
        evidence_text = str(triplet.get("evidence_text") or "").strip()

        if _has_candidate_cue(evidence_text):
            if predicate in {"is", "constitutes"} and _is_dark_matter_term(subject):
                notes.append(
                    f"downgraded candidate-like triplet to candidate_for: subject={subject!r}, object={obj!r}"
                )
                kept.append(
                    {
                        "subject": obj,
                        "predicate": "candidate_for",
                        "object": "dark matter",
                        "claim_type": "candidate",
                        "evidence_text": evidence_text,
                    }
                )
                continue
            if predicate == "constitutes" and _is_dark_matter_term(obj):
                notes.append(
                    f"downgraded candidate-like constitutes triplet to candidate_for: subject={subject!r}, object={obj!r}"
                )
                kept.append(
                    {
                        "subject": subject,
                        "predicate": "candidate_for",
                        "object": "dark matter",
                        "claim_type": "candidate",
                        "evidence_text": evidence_text,
                    }
                )
                continue

        if predicate == "constitutes" and not _is_dark_matter_term(obj):
            notes.append(
                f"filtered invalid constitutes triplet: subject={subject!r}, object={obj!r}"
            )
            continue

        if predicate == "composed_of" and not _is_dark_matter_term(subject):
            notes.append(
                f"filtered invalid composed_of triplet: subject={subject!r}, object={obj!r}"
            )
            continue

        if predicate == "is" and _is_dark_matter_term(subject) and _looks_like_system_level_object(obj):
            notes.append(
                f"filtered likely non-ontological identity triplet: subject={subject!r}, object={obj!r}"
            )
            continue

        if predicate == "is" and _is_dark_matter_term(subject) and _is_generic_identity_object(obj):
            notes.append(
                f"filtered underspecified identity triplet: subject={subject!r}, object={obj!r}"
            )
            continue

        if predicate == "constitutes" and claim_type == "constitution" and _has_candidate_cue(evidence_text):
            notes.append(
                f"filtered ambiguous constitutes triplet with candidate cue: subject={subject!r}, object={obj!r}"
            )
            continue

        kept.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "claim_type": claim_type,
                "evidence_text": evidence_text,
            }
        )

    return kept, notes


def flatten_triplet_requests(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(columns=["bibcode", "year", "title", "subject", "predicate", "object", "evidence_text"])

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            for triplet in payload.get("triplets", []):
                rows.append(
                    {
                        "bibcode": payload.get("bibcode"),
                        "year": payload.get("year"),
                        "title": payload.get("title"),
                        "subject": triplet.get("subject"),
                        "predicate": triplet.get("predicate"),
                        "object": triplet.get("object"),
                        "claim_type": triplet.get("claim_type"),
                        "evidence_text": triplet.get("evidence_text"),
                    }
                )
    return pd.DataFrame(rows)


def write_run_manifest(
    path: Path,
    *,
    run_id: str,
    model: str,
    abstract_mode: str,
    input_path: Path,
    prompt_sha256: str,
    n_input_rows: int,
    n_completed_rows: int,
    n_flat_triplets: int,
    extra: dict[str, Any] | None = None,
) -> None:
    manifest = {
        "run_id": run_id,
        "model": model,
        "abstract_mode": abstract_mode,
        "input_path": str(input_path),
        "prompt_sha256": prompt_sha256,
        "n_input_rows": int(n_input_rows),
        "n_completed_rows": int(n_completed_rows),
        "n_flat_triplets": int(n_flat_triplets),
    }
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
