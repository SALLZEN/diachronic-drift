from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dm_term_normalization.tfidf_preproc import preproc, tokenize_preproc_text


CATEGORY_CONFIG = {
    "astrophysics": {
        "display": "Astrophysics",
        "count_col": "Astrophysics",
        "prob_col": "p_astro",
        "log_odds_col": "log_odds_astro",
        "folder": "discourse_astrophysics",
    },
    "high-energy physics": {
        "display": "High-energy physics",
        "count_col": "High-energy physics",
        "prob_col": "p_hep",
        "log_odds_col": "log_odds_hep",
        "folder": "discourse_high_energy_physics",
    },
}


GENERIC_TERMS = {
    "dark matter",
    "dark-matter",
    "lambda-cold",
    "lambda-cold dark-matter",
    "dark",
    "matter",
    "model",
    "models",
    "paper",
    "study",
    "studies",
    "data",
    "result",
    "results",
    "analysis",
    "approach",
    "method",
    "methods",
    "problem",
    "problems",
    "effect",
    "effects",
    "case",
    "cases",
    "work",
    "mass",
    "energy",
    "time",
    "space",
    "system",
    "systems",
    "universe",
    "parameter",
    "parameters",
    "constraint",
    "constraints",
    "observation",
    "observations",
    "evidence",
    "question",
    "possibility",
    "possible",
    "new",
    "recent",
    "based",
    "using",
    "however",
    "investigate",
    "consider",
    "compare",
    "compared",
    "produced",
    "future",
    "prediction",
    "predictions",
    "test",
    "tests",
    "state",
    "states",
    "value",
    "values",
    "set",
    "different",
    "physics",
    "properties",
    "property",
    "component",
    "components",
    "scenario",
    "scenarios",
    "effective",
    "total",
    "local",
    "times",
    "required",
    "relevant",
    "additional",
    "existing",
    "presence",
    "associated",
    "derived",
    "upper",
    "near",
    "naturally",
    "given",
    "obtain",
    "calculate",
    "examine",
    "explain",
    "suggest",
    "proposed",
    "lead",
    "leads",
    "include",
    "included",
    "production",
    "extended",
    "experimental",
    "viable",
}

WEAK_PARTS = {
    "new",
    "possible",
    "different",
    "various",
    "based",
    "using",
    "recent",
    "present",
    "show",
    "find",
    "study",
    "result",
    "however",
    "consider",
    "investigate",
    "compare",
    "compared",
    "produced",
    "future",
    "proposed",
    "given",
    "derived",
    "existing",
    "additional",
    "relevant",
    "viable",
}

BAD_PARTS = {
    "sub",
    "sup",
    "math",
    "token",
    "eq",
    "fig",
    "mml",
    "mrow",
    "mo",
    "mi",
    "mn",
    "msub",
    "msup",
    "msubsup",
}


def load_wordlist(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def build_stopwords(sklearn_stopwords_path: Path, final_stopwords_path: Path) -> set[str]:
    stopwords = load_wordlist(sklearn_stopwords_path) | load_wordlist(final_stopwords_path)
    stopwords |= {"math", "token"}
    return stopwords


def is_substantive(term: str) -> bool:
    if not isinstance(term, str):
        return False
    tokens = term.split()
    if all(re.match(r"^[\d\.\-\+]+$", token) for token in tokens):
        return False
    if all(re.match(r"^\d{2,4}$", token) for token in tokens):
        return False
    return True


def is_valid_term(term: str, stopwords: set[str]) -> bool:
    if not isinstance(term, str):
        return False
    if term in GENERIC_TERMS:
        return False
    if not is_substantive(term):
        return False

    parts = term.split()
    if any(part in BAD_PARTS for part in parts):
        return False

    if len(parts) == 1:
        token = parts[0]
        if token in stopwords or len(token) < 4:
            return False
    else:
        if any(part in WEAK_PARTS for part in parts):
            return False
        if sum(part not in stopwords for part in parts) < 2:
            return False
    return True


def safe_npmi(
    joint_count: np.ndarray,
    source_freq: np.ndarray,
    target_freq: np.ndarray,
    n_docs: int | np.ndarray,
) -> np.ndarray:
    p_xy = joint_count / n_docs
    p_x = source_freq / n_docs
    p_y = target_freq / n_docs
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(p_xy / (p_x * p_y)) / -np.log(p_xy)
    out[~np.isfinite(out)] = np.nan
    return out


def term_contains(shorter: str, longer: str) -> bool:
    short_tokens = shorter.split()
    long_tokens = longer.split()
    if len(short_tokens) >= len(long_tokens):
        return False
    for idx in range(len(long_tokens) - len(short_tokens) + 1):
        if long_tokens[idx : idx + len(short_tokens)] == short_tokens:
            return True
    return False


def normalize_text_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "|".join(sorted(set(parts)))
    text = str(value).strip()
    return text


def normalize_arxiv_class_value(value: Any) -> str:
    text = normalize_text_value(value)
    return text if text else "MISSING"


def dominant_label(values: list[str]) -> tuple[str, float]:
    if not values:
        return "MISSING", float("nan")
    counts = Counter(values)
    label, n = max(counts.items(), key=lambda item: (item[1], item[0]))
    return label, n / len(values)


def pipe_join_unique(values: list[str]) -> str:
    clean = sorted({value for value in values if value and value != "MISSING"})
    if not clean:
        return "MISSING"
    return "|".join(clean)


def summarize_doc_ids(doc_ids: np.ndarray, years: np.ndarray, arxiv_classes: np.ndarray) -> dict[str, object]:
    if len(doc_ids) == 0:
        return {
            "year_min": pd.NA,
            "year_max": pd.NA,
            "year_mean": np.nan,
            "year_median": np.nan,
            "arxiv_class": pd.NA,
            "dominant_arxiv_class": pd.NA,
            "dominant_arxiv_class_share": np.nan,
        }

    year_slice = years[doc_ids]
    class_values = [normalize_arxiv_class_value(value) for value in arxiv_classes[doc_ids].tolist()]
    dominant_class, dominant_share = dominant_label(class_values)
    return {
        "year_min": int(year_slice.min()),
        "year_max": int(year_slice.max()),
        "year_mean": float(year_slice.mean()),
        "year_median": float(np.median(year_slice)),
        "arxiv_class": pipe_join_unique(class_values),
        "dominant_arxiv_class": dominant_class,
        "dominant_arxiv_class_share": float(dominant_share),
    }


def count_categories_for_doc_ids(doc_ids: np.ndarray, categories: np.ndarray) -> tuple[int, int]:
    if len(doc_ids) == 0:
        return 0, 0
    values = categories[doc_ids]
    astro = int(np.sum(values == "astrophysics"))
    hep = int(np.sum(values == "high-energy physics"))
    return astro, hep


def compute_share_metrics(
    astro_count: int,
    hep_count: int,
    astro_total: int,
    hep_total: int,
    epsilon: float = 0.5,
) -> dict[str, float]:
    share_astro = float(astro_count / astro_total) if astro_total else np.nan
    share_hep = float(hep_count / hep_total) if hep_total else np.nan

    astro_smoothed = (astro_count + epsilon) / (astro_total + 2 * epsilon) if astro_total else np.nan
    hep_smoothed = (hep_count + epsilon) / (hep_total + 2 * epsilon) if hep_total else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_share_ratio = np.log2(hep_smoothed / astro_smoothed)
    if not np.isfinite(log_share_ratio):
        log_share_ratio = np.nan

    return {
        "share_astro": share_astro,
        "share_hep": share_hep,
        "log_share_ratio": float(log_share_ratio) if pd.notna(log_share_ratio) else np.nan,
    }


def to_python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def infer_graphml_type(values: list[Any]) -> str:
    for value in values:
        value = to_python_scalar(value)
        if value is None:
            continue
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "long"
        if isinstance(value, float):
            return "double"
        return "string"
    return "string"


def graphml_string(value: Any, value_type: str) -> str:
    value = to_python_scalar(value)
    if value is None:
        return ""
    if value_type == "boolean":
        return "true" if bool(value) else "false"
    return str(value)


def write_graphml(nodes: pd.DataFrame, edges: pd.DataFrame, path: Path, graph_id: str) -> None:
    graphml_ns = "http://graphml.graphdrawing.org/xmlns"
    xsi_ns = "http://www.w3.org/2001/XMLSchema-instance"
    schema_loc = (
        "http://graphml.graphdrawing.org/xmlns "
        "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
    )
    ET.register_namespace("", graphml_ns)
    ET.register_namespace("xsi", xsi_ns)

    root = ET.Element(
        f"{{{graphml_ns}}}graphml",
        {f"{{{xsi_ns}}}schemaLocation": schema_loc},
    )

    node_columns = list(nodes.columns)
    edge_columns = list(edges.columns)
    key_specs: list[tuple[str, str, str, str]] = []
    key_id = 0
    for column in node_columns:
        value_type = infer_graphml_type(nodes[column].tolist())
        key_name = f"d{key_id}"
        key_id += 1
        key_specs.append((key_name, "node", column, value_type))
    for column in edge_columns:
        value_type = infer_graphml_type(edges[column].tolist())
        key_name = f"d{key_id}"
        key_id += 1
        key_specs.append((key_name, "edge", column, value_type))

    for key_name, target, attr_name, attr_type in key_specs:
        ET.SubElement(
            root,
            f"{{{graphml_ns}}}key",
            {
                "id": key_name,
                "for": target,
                "attr.name": attr_name,
                "attr.type": attr_type,
            },
        )

    graph = ET.SubElement(
        root,
        f"{{{graphml_ns}}}graph",
        {"id": graph_id, "edgedefault": "undirected"},
    )

    node_map = {node_id: f"n{idx}" for idx, node_id in enumerate(nodes["Id"].tolist())}
    node_keys = {spec[2]: (spec[0], spec[3]) for spec in key_specs if spec[1] == "node"}
    edge_keys = {spec[2]: (spec[0], spec[3]) for spec in key_specs if spec[1] == "edge"}

    for row in nodes.to_dict(orient="records"):
        node = ET.SubElement(graph, f"{{{graphml_ns}}}node", {"id": node_map[row["Id"]]})
        for column in node_columns:
            key_name, value_type = node_keys[column]
            value = to_python_scalar(row[column])
            if value is None:
                continue
            data = ET.SubElement(node, f"{{{graphml_ns}}}data", {"key": key_name})
            data.text = graphml_string(value, value_type)

    for idx, row in enumerate(edges.to_dict(orient="records")):
        edge = ET.SubElement(
            graph,
            f"{{{graphml_ns}}}edge",
            {
                "id": f"e{idx}",
                "source": node_map[row["Source"]],
                "target": node_map[row["Target"]],
            },
        )
        for column in edge_columns:
            key_name, value_type = edge_keys[column]
            value = to_python_scalar(row[column])
            if value is None:
                continue
            data = ET.SubElement(edge, f"{{{graphml_ns}}}data", {"key": key_name})
            data.text = graphml_string(value, value_type)

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def load_merged_papers(
    ads_path: Path,
    dm_models_path: Path,
    selected_categories: tuple[str, ...],
    max_docs: int | None = None,
) -> pd.DataFrame:
    ads = pd.read_parquet(
        ads_path,
        columns=["bibcode", "year", "abstract", "arxiv_class", "arxiv_category"],
    )
    dm = pd.read_parquet(
        dm_models_path,
        columns=["bibcode", "primary_arxiv_class", "arxiv_category", "abstract_clean"],
    )

    papers = ads.merge(dm, on="bibcode", how="left", suffixes=("_ads", "_dm"))
    papers["network_text"] = papers["abstract_clean"].fillna("").astype(str)
    needs_fallback = papers["network_text"].str.strip().eq("")
    papers["text_source"] = np.where(needs_fallback, "ads_abstract", "dm_models_abstract_clean")
    papers.loc[needs_fallback, "network_text"] = (
        papers.loc[needs_fallback, "abstract"].fillna("").astype(str)
    )

    papers["merged_arxiv_class"] = papers["primary_arxiv_class"].apply(normalize_arxiv_class_value)
    missing_class = papers["merged_arxiv_class"].eq("MISSING")
    papers.loc[missing_class, "merged_arxiv_class"] = papers.loc[missing_class, "arxiv_class"].apply(
        normalize_arxiv_class_value
    )
    papers["merged_arxiv_category"] = papers["arxiv_category_dm"].fillna("").astype(str).str.strip()
    missing_category = papers["merged_arxiv_category"].eq("")
    papers.loc[missing_category, "merged_arxiv_category"] = (
        papers.loc[missing_category, "arxiv_category_ads"].fillna("").astype(str).str.strip()
    )

    papers["year"] = pd.to_numeric(papers["year"], errors="coerce")
    papers = papers[papers["year"].notna()].copy()
    papers["year"] = papers["year"].astype(int)
    papers = papers[papers["network_text"].str.strip().ne("")].copy()
    papers = papers[papers["merged_arxiv_category"].isin(selected_categories)].copy()

    if max_docs is not None:
        papers = papers.iloc[:max_docs].copy()

    return papers.reset_index(drop=True)


def load_keyness_candidates(
    keyness_unigrams_path: Path,
    keyness_bigrams_path: Path,
    stopwords: set[str],
    min_total_df: int,
    top_unigrams_per_category: int,
    top_bigrams_per_category: int,
) -> pd.DataFrame:
    configs = [
        (pd.read_csv(keyness_unigrams_path), "keyness_unigrams", "unigram", top_unigrams_per_category),
        (pd.read_csv(keyness_bigrams_path), "keyness_bigrams", "bigram", top_bigrams_per_category),
    ]

    frames: list[pd.DataFrame] = []
    for table, source_table, term_type, limit in configs:
        table = table.copy()
        table = table[table["term"].notna()].copy()
        table["term"] = table["term"].astype(str).str.strip().str.lower()
        table = table[table["term"].ne("") & table["term"].ne("nan")].copy()
        table = table[table["total_df"] >= min_total_df].copy()
        table = table[table["term"].map(lambda value: is_valid_term(value, stopwords))].copy()

        for category, config in CATEGORY_CONFIG.items():
            field_table = table[table[config["log_odds_col"]] > 0].copy()
            field_table["arxiv_category"] = category
            field_table["source_table"] = source_table
            field_table["term_type"] = term_type
            field_table["field_doc_freq"] = field_table[config["count_col"]]
            field_table["field_probability"] = field_table[config["prob_col"]]
            field_table["field_log_odds"] = field_table[config["log_odds_col"]]
            field_table = field_table.sort_values(
                ["salience", "total_df", "term"],
                ascending=[False, False, True],
            ).head(limit)
            frames.append(field_table)

    selected = pd.concat(frames, ignore_index=True)
    selected = selected.sort_values(
        ["arxiv_category", "salience", "total_df", "term"],
        ascending=[True, False, False, True],
    )
    selected = selected.drop_duplicates(["arxiv_category", "term"], keep="first").reset_index(drop=True)
    return selected


def build_candidate_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.groupby(["arxiv_category", "term_type"], as_index=False)
        .agg(
            terms=("term", "nunique"),
            salience_min=("salience", "min"),
            salience_max=("salience", "max"),
        )
        .sort_values(["arxiv_category", "term_type"])
        .reset_index(drop=True)
    )


def build_term_matrix(texts: list[str], terms: list[str]) -> tuple[np.ndarray, np.ndarray, CountVectorizer]:
    if not terms:
        raise ValueError("Cannot build a term matrix without at least one candidate term.")

    unique_terms = list(dict.fromkeys(terms))
    max_ngram = max(len(term.split()) for term in unique_terms)
    vectorizer = CountVectorizer(
        preprocessor=preproc,
        tokenizer=tokenize_preproc_text,
        token_pattern=None,
        lowercase=False,
        binary=True,
        vocabulary=unique_terms,
        ngram_range=(1, max_ngram),
    )
    matrix = vectorizer.fit_transform(texts).astype(np.int32)
    features = vectorizer.get_feature_names_out()
    return matrix, features, vectorizer


def export_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=json_default)


def build_field_specific_network(
    papers: pd.DataFrame,
    candidates: pd.DataFrame,
    export_root: Path,
    max_docs: int | None,
    min_edge_doc_share: float,
    min_edge_count_floor: int,
    min_edge_npmi: float,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for category, config in CATEGORY_CONFIG.items():
        category_papers = papers[papers["merged_arxiv_category"] == category].copy()
        category_candidates = candidates[candidates["arxiv_category"] == category].copy()
        category_terms = category_candidates["term"].tolist()
        if not category_terms:
            raise ValueError(f"No candidate terms available for category {category!r}.")

        matrix, terms, _ = build_term_matrix(category_papers["network_text"].tolist(), category_terms)
        freq = np.asarray(matrix.sum(axis=0)).ravel()
        keep_mask = freq > 0
        matrix = matrix[:, keep_mask]
        terms = terms[keep_mask]
        freq = freq[keep_mask]
        if len(terms) == 0:
            raise ValueError(f"No candidate terms survived frequency filtering for category {category!r}.")

        category_candidates = (
            category_candidates.set_index("term").loc[terms].reset_index().rename(columns={"index": "term"})
        )

        years = category_papers["year"].to_numpy(dtype=np.int32)
        arxiv_classes = category_papers["merged_arxiv_class"].to_numpy(dtype=object)
        n_docs = len(category_papers)
        min_edge_count = max(min_edge_count_floor, math.ceil(n_docs * min_edge_doc_share))

        cooc = (matrix.T @ matrix).tocoo()
        mask = cooc.row < cooc.col
        rows = cooc.row[mask]
        cols = cooc.col[mask]
        counts = cooc.data[mask].astype(np.int32)
        npmi = safe_npmi(
            counts.astype(float),
            freq[rows].astype(float),
            freq[cols].astype(float),
            n_docs,
        )

        edges = pd.DataFrame(
            {
                "Source": terms[rows],
                "Target": terms[cols],
                "count": counts,
                "npmi": npmi,
                "source_doc_freq": freq[rows],
                "target_doc_freq": freq[cols],
            }
        )
        edges = edges[np.isfinite(edges["npmi"])].copy()
        edges = edges[(edges["count"] >= min_edge_count) & (edges["npmi"] >= min_edge_npmi)].copy()
        containment_mask = edges.apply(
            lambda row: term_contains(row["Source"], row["Target"])
            or term_contains(row["Target"], row["Source"]),
            axis=1,
        )
        edges = edges[~containment_mask].copy().reset_index(drop=True)

        matrix_csc = matrix.tocsc()
        postings = {
            terms[idx]: matrix_csc.indices[matrix_csc.indptr[idx] : matrix_csc.indptr[idx + 1]]
            for idx in range(len(terms))
        }

        edge_meta_rows = []
        for edge_row in edges.itertuples(index=False):
            doc_ids = np.intersect1d(postings[edge_row.Source], postings[edge_row.Target], assume_unique=True)
            edge_meta_rows.append(summarize_doc_ids(doc_ids, years, arxiv_classes))
        if edge_meta_rows:
            edges = pd.concat([edges, pd.DataFrame(edge_meta_rows)], axis=1)
        else:
            for column, value in summarize_doc_ids(np.array([], dtype=np.int32), years, arxiv_classes).items():
                edges[column] = value
        edges["arxiv_category"] = category
        edges["Weight"] = edges["count"]
        edges = edges.sort_values(["count", "npmi"], ascending=[False, False]).reset_index(drop=True)

        node_ids = sorted(set(edges["Source"]) | set(edges["Target"])) if not edges.empty else list(terms)
        node_meta_rows = []
        for term in node_ids:
            doc_ids = postings[term]
            row = {"Id": term, **summarize_doc_ids(doc_ids, years, arxiv_classes)}
            node_meta_rows.append(row)
        node_meta = pd.DataFrame(node_meta_rows)
        node_stats = category_candidates.set_index("term").loc[node_ids].reset_index()
        node_stats = node_stats.rename(columns={"term": "Id"})
        node_stats["Label"] = node_stats["Id"]

        if edges.empty:
            node_stats["degree"] = 0
        else:
            degree = (
                pd.concat(
                    [
                        edges[["Source"]].rename(columns={"Source": "Id"}),
                        edges[["Target"]].rename(columns={"Target": "Id"}),
                    ],
                    ignore_index=True,
                )
                .value_counts("Id")
                .reindex(node_stats["Id"], fill_value=0)
                .to_numpy()
            )
            node_stats["degree"] = degree

        term_freq_map = {term: int(term_freq) for term, term_freq in zip(terms.tolist(), freq.tolist())}
        node_stats["doc_freq_in_network"] = node_stats["Id"].map(term_freq_map).astype(int)
        node_stats["doc_share_in_network"] = node_stats["doc_freq_in_network"] / n_docs
        node_stats = node_stats.merge(node_meta, on="Id", how="left")
        node_stats["arxiv_category"] = category
        node_stats = node_stats[
            [
                "Id",
                "Label",
                "term_type",
                "source_table",
                "arxiv_category",
                "arxiv_class",
                "dominant_arxiv_class",
                "dominant_arxiv_class_share",
                "total_df",
                config["count_col"],
                config["prob_col"],
                config["log_odds_col"],
                "field_doc_freq",
                "field_probability",
                "field_log_odds",
                "salience",
                "doc_freq_in_network",
                "doc_share_in_network",
                "degree",
                "year_min",
                "year_max",
                "year_mean",
                "year_median",
            ]
        ].sort_values(["degree", "salience", "Id"], ascending=[False, False, True]).reset_index(drop=True)

        folder_name = config["folder"] if max_docs is None else f"{config['folder']}__sample_{max_docs}"
        export_dir = export_root / folder_name
        export_dir.mkdir(parents=True, exist_ok=True)
        nodes_out = export_dir / "nodes.csv"
        edges_out = export_dir / "edges.csv"
        graphml_out = export_dir / "network.graphml"
        manifest_out = export_dir / "manifest.json"

        node_stats.to_csv(nodes_out, index=False)
        edges.to_csv(edges_out, index=False)
        write_graphml(node_stats, edges, graphml_out, graph_id=folder_name)

        manifest = {
            "network_id": folder_name,
            "graph_type": "field_specific_discourse_network",
            "arxiv_category": category,
            "n_documents": int(n_docs),
            "candidate_terms": int(len(category_candidates)),
            "retained_nodes": int(len(node_stats)),
            "retained_edges": int(len(edges)),
            "min_edge_doc_share": float(min_edge_doc_share),
            "min_edge_count": int(min_edge_count),
            "min_edge_npmi": float(min_edge_npmi),
            "max_docs": None if max_docs is None else int(max_docs),
            "outputs": {
                "nodes_csv": str(nodes_out),
                "edges_csv": str(edges_out),
                "graphml": str(graphml_out),
            },
        }
        export_manifest(manifest_out, manifest)

        results[category] = {
            "export_dir": export_dir,
            "nodes": node_stats,
            "edges": edges,
            "manifest": manifest,
        }

    return results


def build_bipartite_network(
    papers: pd.DataFrame,
    candidates: pd.DataFrame,
    export_root: Path,
    max_docs: int | None,
) -> dict[str, Any]:
    selected_terms = (
        candidates.sort_values(["salience", "total_df", "term"], ascending=[False, False, True])
        .drop_duplicates("term", keep="first")
        .reset_index(drop=True)
    )
    matrix, terms, _ = build_term_matrix(papers["network_text"].tolist(), selected_terms["term"].tolist())
    freq = np.asarray(matrix.sum(axis=0)).ravel()
    keep_mask = freq > 0
    matrix = matrix[:, keep_mask]
    terms = terms[keep_mask]
    freq = freq[keep_mask]
    selected_terms = selected_terms.set_index("term").loc[terms].reset_index().rename(columns={"index": "term"})

    years = papers["year"].to_numpy(dtype=np.int32)
    arxiv_classes = papers["merged_arxiv_class"].to_numpy(dtype=object)
    matrix_csc = matrix.tocsc()
    postings = {
        terms[idx]: matrix_csc.indices[matrix_csc.indptr[idx] : matrix_csc.indptr[idx + 1]]
        for idx in range(len(terms))
    }

    term_nodes = []
    for term in selected_terms["term"]:
        doc_ids = postings[term]
        term_row = selected_terms[selected_terms["term"] == term].iloc[0].to_dict()
        term_nodes.append(
            {
                "Id": term,
                "Label": term,
                "node_type": "term",
                "term_type": term_row["term_type"],
                "source_table": term_row["source_table"],
                "total_df": int(term_row["total_df"]),
                "Astrophysics": int(term_row["Astrophysics"]),
                "High-energy physics": int(term_row["High-energy physics"]),
                "p_astro": float(term_row["p_astro"]),
                "p_hep": float(term_row["p_hep"]),
                "log_odds_astro": float(term_row["log_odds_astro"]),
                "log_odds_hep": float(term_row["log_odds_hep"]),
                "salience": float(term_row["salience"]),
                "dominant_category": term_row["arxiv_category"],
                "dominant_log_odds": float(term_row["field_log_odds"]),
                **summarize_doc_ids(doc_ids, years, arxiv_classes),
            }
        )

    category_nodes = [
        {
            "Id": category,
            "Label": CATEGORY_CONFIG[category]["display"],
            "node_type": "category",
            "term_type": pd.NA,
            "source_table": pd.NA,
            "total_df": pd.NA,
            "Astrophysics": pd.NA,
            "High-energy physics": pd.NA,
            "p_astro": pd.NA,
            "p_hep": pd.NA,
            "log_odds_astro": pd.NA,
            "log_odds_hep": pd.NA,
            "salience": pd.NA,
            "dominant_category": category,
            "dominant_log_odds": pd.NA,
            "arxiv_class": pd.NA,
            "dominant_arxiv_class": pd.NA,
            "dominant_arxiv_class_share": pd.NA,
            "year_min": pd.NA,
            "year_max": pd.NA,
            "year_mean": pd.NA,
            "year_median": pd.NA,
        }
        for category in CATEGORY_CONFIG
    ]

    nodes = pd.concat([pd.DataFrame(term_nodes), pd.DataFrame(category_nodes)], ignore_index=True)

    edge_rows = []
    for row in selected_terms.to_dict(orient="records"):
        edge_rows.append(
            {
                "Source": row["term"],
                "Target": row["arxiv_category"],
                "Weight": float(row["salience"]),
                "salience": float(row["salience"]),
                "term_type": row["term_type"],
                "source_table": row["source_table"],
                "Astrophysics": int(row["Astrophysics"]),
                "High-energy physics": int(row["High-energy physics"]),
                "p_astro": float(row["p_astro"]),
                "p_hep": float(row["p_hep"]),
                "log_odds_astro": float(row["log_odds_astro"]),
                "log_odds_hep": float(row["log_odds_hep"]),
            }
        )
    edges = pd.DataFrame(edge_rows).sort_values(["Weight", "Source"], ascending=[False, True]).reset_index(drop=True)

    folder_name = "term_category_bipartite" if max_docs is None else f"term_category_bipartite__sample_{max_docs}"
    export_dir = export_root / folder_name
    export_dir.mkdir(parents=True, exist_ok=True)
    nodes_out = export_dir / "nodes.csv"
    edges_out = export_dir / "edges.csv"
    graphml_out = export_dir / "network.graphml"
    manifest_out = export_dir / "manifest.json"

    nodes.to_csv(nodes_out, index=False)
    edges.to_csv(edges_out, index=False)
    write_graphml(nodes, edges, graphml_out, graph_id=folder_name)

    manifest = {
        "network_id": folder_name,
        "graph_type": "term_category_bipartite",
        "n_documents": int(len(papers)),
        "retained_term_nodes": int(len(term_nodes)),
        "category_nodes": int(len(category_nodes)),
        "retained_edges": int(len(edges)),
        "max_docs": None if max_docs is None else int(max_docs),
        "outputs": {
            "nodes_csv": str(nodes_out),
            "edges_csv": str(edges_out),
            "graphml": str(graphml_out),
        },
    }
    export_manifest(manifest_out, manifest)

    return {
        "export_dir": export_dir,
        "nodes": nodes,
        "edges": edges,
        "manifest": manifest,
    }


def build_dark_matter_induced_network(
    papers: pd.DataFrame,
    candidates: pd.DataFrame,
    export_root: Path,
    max_docs: int | None,
    min_edge_doc_share: float,
    min_edge_count_floor: int,
    min_edge_npmi: float,
    anchor_term: str = "dark matter",
) -> dict[str, Any]:
    mask = papers["network_text"].str.contains(anchor_term, case=False, regex=False, na=False)
    subset = papers[mask].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No papers matched the literal phrase {anchor_term!r}.")

    selected_terms = (
        candidates.sort_values(["salience", "total_df", "term"], ascending=[False, False, True])
        .drop_duplicates("term", keep="first")
        .reset_index(drop=True)
    )
    vocabulary = [anchor_term, *selected_terms["term"].tolist()]
    matrix, terms, _ = build_term_matrix(subset["network_text"].tolist(), vocabulary)
    freq = np.asarray(matrix.sum(axis=0)).ravel()
    keep_mask = (freq > 0) | (terms == anchor_term)
    matrix = matrix[:, keep_mask]
    terms = terms[keep_mask]
    freq = freq[keep_mask]

    term_meta = selected_terms.set_index("term").reindex([term for term in terms if term != anchor_term]).reset_index()
    term_meta = term_meta.rename(columns={"index": "term"})

    years = subset["year"].to_numpy(dtype=np.int32)
    arxiv_classes = subset["merged_arxiv_class"].to_numpy(dtype=object)
    categories = subset["merged_arxiv_category"].to_numpy(dtype=object)
    n_docs = len(subset)
    astro_total = int(np.sum(categories == "astrophysics"))
    hep_total = int(np.sum(categories == "high-energy physics"))
    min_edge_count = max(min_edge_count_floor, math.ceil(n_docs * min_edge_doc_share))

    cooc = (matrix.T @ matrix).tocoo()
    mask = cooc.row < cooc.col
    rows = cooc.row[mask]
    cols = cooc.col[mask]
    counts = cooc.data[mask].astype(np.int32)
    npmi = safe_npmi(
        counts.astype(float),
        freq[rows].astype(float),
        freq[cols].astype(float),
        n_docs,
    )

    edges = pd.DataFrame(
        {
            "Source": terms[rows],
            "Target": terms[cols],
            "count_total": counts,
            "npmi_total": npmi,
            "source_doc_freq": freq[rows],
            "target_doc_freq": freq[cols],
        }
    )
    edges = edges[np.isfinite(edges["npmi_total"])].copy()
    edges = edges[(edges["count_total"] >= min_edge_count) & (edges["npmi_total"] >= min_edge_npmi)].copy()
    containment_mask = edges.apply(
        lambda row: (
            row["Source"] != anchor_term
            and row["Target"] != anchor_term
            and (
                term_contains(row["Source"], row["Target"])
                or term_contains(row["Target"], row["Source"])
            )
        ),
        axis=1,
    )
    edges = edges[~containment_mask].copy().reset_index(drop=True)

    matrix_csc = matrix.tocsc()
    postings = {
        terms[idx]: matrix_csc.indices[matrix_csc.indptr[idx] : matrix_csc.indptr[idx + 1]]
        for idx in range(len(terms))
    }

    edge_meta_rows = []
    for edge_row in edges.itertuples(index=False):
        doc_ids = np.intersect1d(postings[edge_row.Source], postings[edge_row.Target], assume_unique=True)
        astro_count, hep_count = count_categories_for_doc_ids(doc_ids, categories)
        edge_meta = summarize_doc_ids(doc_ids, years, arxiv_classes)
        edge_meta.update(
            {
                "count_astro": astro_count,
                "count_hep": hep_count,
                **compute_share_metrics(astro_count, hep_count, astro_total, hep_total),
            }
        )
        edge_meta_rows.append(edge_meta)

    if edge_meta_rows:
        edges = pd.concat([edges, pd.DataFrame(edge_meta_rows)], axis=1)
    else:
        empty_meta = summarize_doc_ids(np.array([], dtype=np.int32), years, arxiv_classes)
        empty_meta.update(
            {
                "count_astro": 0,
                "count_hep": 0,
                "share_astro": np.nan,
                "share_hep": np.nan,
                "log_share_ratio": np.nan,
            }
        )
        for column, value in empty_meta.items():
            edges[column] = value

    star_rows = []
    for term in terms.tolist():
        if term == anchor_term:
            continue
        doc_ids = postings[term]
        if len(doc_ids) == 0:
            continue
        astro_count, hep_count = count_categories_for_doc_ids(doc_ids, categories)
        star_meta = summarize_doc_ids(doc_ids, years, arxiv_classes)
        star_meta.update(
            {
                "count_astro": astro_count,
                "count_hep": hep_count,
                **compute_share_metrics(astro_count, hep_count, astro_total, hep_total),
            }
        )
        star_rows.append(
            {
                "Source": anchor_term,
                "Target": term,
                "count_total": int(len(doc_ids)),
                "npmi_total": 0.0,
                "source_doc_freq": int(n_docs),
                "target_doc_freq": int(len(doc_ids)),
                **star_meta,
            }
        )

    star_edges = pd.DataFrame(star_rows)
    edges = edges[
        ~(
            ((edges["Source"] == anchor_term) & (edges["Target"] != anchor_term))
            | ((edges["Target"] == anchor_term) & (edges["Source"] != anchor_term))
        )
    ].copy()
    if not star_edges.empty:
        edges = pd.concat([edges, star_edges], ignore_index=True)

    edges["Weight"] = edges["count_total"]
    edges = edges[
        [
            "Source",
            "Target",
            "count_total",
            "count_astro",
            "count_hep",
            "share_astro",
            "share_hep",
            "log_share_ratio",
            "npmi_total",
            "source_doc_freq",
            "target_doc_freq",
            "arxiv_class",
            "dominant_arxiv_class",
            "dominant_arxiv_class_share",
            "year_min",
            "year_max",
            "year_mean",
            "year_median",
            "Weight",
        ]
    ].sort_values(["count_total", "npmi_total"], ascending=[False, False]).reset_index(drop=True)

    node_ids = sorted(set(edges["Source"]) | set(edges["Target"])) if not edges.empty else terms.tolist()
    term_meta_map = {
        row["term"]: row
        for row in term_meta.to_dict(orient="records")
    }
    node_rows = []
    term_freq_map = {term: int(term_freq) for term, term_freq in zip(terms.tolist(), freq.tolist())}
    for term in node_ids:
        doc_ids = postings[term]
        astro_count, hep_count = count_categories_for_doc_ids(doc_ids, categories)
        node_meta = summarize_doc_ids(doc_ids, years, arxiv_classes)
        node_meta.update(compute_share_metrics(astro_count, hep_count, astro_total, hep_total))
        base = term_meta_map.get(
            term,
            {
                "term_type": "anchor",
                "source_table": "literal_anchor",
                "total_df": term_freq_map[term],
                "Astrophysics": astro_count,
                "High-energy physics": hep_count,
                "p_astro": np.nan,
                "p_hep": np.nan,
                "log_odds_astro": np.nan,
                "log_odds_hep": np.nan,
                "salience": np.nan,
                "arxiv_category": "all",
                "field_log_odds": np.nan,
            },
        )
        node_rows.append(
            {
                "Id": term,
                "Label": term,
                "term_type": base["term_type"],
                "source_table": base["source_table"],
                "total_df": int(base["total_df"]) if pd.notna(base["total_df"]) else pd.NA,
                "Astrophysics": int(base["Astrophysics"]) if pd.notna(base["Astrophysics"]) else pd.NA,
                "High-energy physics": int(base["High-energy physics"]) if pd.notna(base["High-energy physics"]) else pd.NA,
                "p_astro": float(base["p_astro"]) if pd.notna(base["p_astro"]) else np.nan,
                "p_hep": float(base["p_hep"]) if pd.notna(base["p_hep"]) else np.nan,
                "log_odds_astro": float(base["log_odds_astro"]) if pd.notna(base["log_odds_astro"]) else np.nan,
                "log_odds_hep": float(base["log_odds_hep"]) if pd.notna(base["log_odds_hep"]) else np.nan,
                "salience": float(base["salience"]) if pd.notna(base["salience"]) else np.nan,
                "node_freq_total": int(len(doc_ids)),
                "node_freq_astro": astro_count,
                "node_freq_hep": hep_count,
                "dominant_category": base["arxiv_category"],
                **node_meta,
            }
        )

    nodes = pd.DataFrame(node_rows)
    if edges.empty:
        nodes["degree"] = 0
    else:
        nodes["degree"] = (
            pd.concat(
                [
                    edges[["Source"]].rename(columns={"Source": "Id"}),
                    edges[["Target"]].rename(columns={"Target": "Id"}),
                ],
                ignore_index=True,
            )
            .value_counts("Id")
            .reindex(nodes["Id"], fill_value=0)
            .to_numpy()
        )
    nodes = nodes[
        [
            "Id",
            "Label",
            "term_type",
            "source_table",
            "total_df",
            "Astrophysics",
            "High-energy physics",
            "p_astro",
            "p_hep",
            "log_odds_astro",
            "log_odds_hep",
            "salience",
            "node_freq_total",
            "node_freq_astro",
            "node_freq_hep",
            "share_astro",
            "share_hep",
            "log_share_ratio",
            "degree",
            "arxiv_class",
            "dominant_arxiv_class",
            "dominant_arxiv_class_share",
            "dominant_category",
            "year_min",
            "year_max",
            "year_mean",
            "year_median",
        ]
    ].sort_values(["node_freq_total", "degree", "Id"], ascending=[False, False, True]).reset_index(drop=True)

    folder_name = "dark_matter_induced_discourse_network"
    if max_docs is not None:
        folder_name = f"{folder_name}__sample_{max_docs}"
    export_dir = export_root / folder_name
    export_dir.mkdir(parents=True, exist_ok=True)
    nodes_out = export_dir / "nodes.csv"
    edges_out = export_dir / "edges.csv"
    graphml_out = export_dir / "network.graphml"
    manifest_out = export_dir / "manifest.json"

    nodes.to_csv(nodes_out, index=False)
    edges.to_csv(edges_out, index=False)
    write_graphml(nodes, edges, graphml_out, graph_id=folder_name)

    manifest = {
        "network_id": folder_name,
        "graph_type": "dark_matter_induced_discourse_network",
        "anchor_term": anchor_term,
        "literal_filter": anchor_term,
        "n_documents": int(n_docs),
        "n_documents_astro": astro_total,
        "n_documents_hep": hep_total,
        "retained_nodes": int(len(nodes)),
        "retained_edges": int(len(edges)),
        "min_edge_doc_share": float(min_edge_doc_share),
        "min_edge_count": int(min_edge_count),
        "min_edge_npmi": float(min_edge_npmi),
        "max_docs": None if max_docs is None else int(max_docs),
        "outputs": {
            "nodes_csv": str(nodes_out),
            "edges_csv": str(edges_out),
            "graphml": str(graphml_out),
        },
    }
    export_manifest(manifest_out, manifest)

    return {
        "export_dir": export_dir,
        "nodes": nodes,
        "edges": edges,
        "manifest": manifest,
        "subset_papers": subset,
    }
