"""
NLP-driven data acquisition for quantum circuit dataset.

Main NLP techniques:
  - Layout-aware paragraph extraction (geometric + semantic filtering)
  - Multi-level text classification (math vs prose, circuit vs bibliography)
  - Semantic relevance scoring (lexicon-based + entity recognition)
  - Position tracking for traceability
"""

import os
import time
import logging
import json
import re
import csv
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import difflib

import fitz  # PyMuPDF
import arxiv

# Internal modules
from circuit_classifier import load_circuit_model
from extract_circuit_images import extract_and_classify_clusters
from text_layout_utils import (
    extract_footer_page_numbers,
    build_full_text,
    extract_paragraphs_around_figure,
    select_description_sentences,
    clean_description_text,
    is_math_like,
    calculate_semantic_relevance,
)
from gate_problem_extractor import extract_problem_and_gates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

PAPER_LIST_FILE = BASE_DIR / "paper_list_4.txt"
PDF_FOLDER = PROJECT_ROOT / "papers_4"
TARGET_JSON_PATH = PROJECT_ROOT / "dataset_4.json"

TARGET_NUM_CIRCUITS = 250

# HELPER FUNCTIONS

def setup_directories() -> None:
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)


def load_paper_list(filename: Path) -> List[str]:
    """Load arXiv IDs from file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"{filename} not found")
        return []


def download_pdf(arxiv_id: str) -> Optional[str]:
    """Download PDF using the arxiv Python API."""
    arxiv_id_norm = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
    out_path = PDF_FOLDER / f"{arxiv_id_norm.replace('/', '_')}.pdf"

    if out_path.exists():
        logger.info(f" -> {arxiv_id_norm}: already downloaded")
        return str(out_path)

    try:
        search = arxiv.Search(id_list=[arxiv_id_norm], max_results=1)

        # arxiv lib version handling
        try:
            result = next(search.results())
        except Exception:
            client = arxiv.Client()
            result = next(client.results(search))

        logger.info(f" -> {arxiv_id_norm}: downloading via arxiv API")
        result.download_pdf(dirpath=str(PDF_FOLDER), filename=out_path.name)

        time.sleep(4)  # politeness

        return str(out_path) if out_path.exists() else None

    except StopIteration:
        logger.warning(f" -> {arxiv_id_norm}: no arXiv result found")
        return None
    except Exception as e:
        logger.warning(f" -> {arxiv_id_norm}: arxiv API download failed: {e}")
        return None


def normalize_figure_number(fig):
    """Normalize figure identifiers to integer-only values."""
    if fig is None:
        return -1
    if isinstance(fig, int):
        return fig
    s = str(fig).lower().strip()
    if re.fullmatch(r"[a-z]", s):
        return -1
    m = re.match(r"(\d+(?:\.\d+)*)([a-z]?)", s)
    if not m:
        return -1
    base = int(m.group(1).replace(".", ""))
    suffix = m.group(2)
    if not suffix:
        return base
    return base * 100 + (ord(suffix) - ord("a") + 1)


# NLP / Text Processing

def _norm(s: str) -> str:
    s = s.replace("-\n", "")
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


def _variants(text: str) -> List[str]:
    v: List[str] = []
    t = text or ""
    t0 = _norm(t)
    if t0:
        v.append(t0)
    c = _norm(clean_description_text(t))
    if c and c not in v:
        v.append(c)
    return v


def find_sentence_position_best(
    full_text: str,
    sentence_index: List[Dict[str, Any]],
    page_idx: int,
    target_text: str,
) -> Optional[Tuple[int, int]]:
    """Find position of sentence in full text."""
    variants = _variants(target_text)
    if not variants:
        return None

    page_sents = [
        s for s in sentence_index
        if s.get("page_index") == page_idx and s.get("text")
    ]
    if not page_sents:
        return None

    # 1) Substring match
    best = None
    best_score = 0.0
    for s in page_sents:
        st = _norm(s["text"])
        if not st:
            continue
        for v in variants:
            if v and v in st:
                score = len(v) / max(len(st), 1)
                if score > best_score:
                    best_score = score
                    best = (s["begin"], s["end"])
    if best and best_score >= 0.35:
        return best

    # 2) Fuzzy match
    best = None
    best_ratio = 0.0
    for s in page_sents:
        st = _norm(s["text"])
        if not st:
            continue
        for v in variants:
            r = difflib.SequenceMatcher(None, v, st).ratio()
            if r > best_ratio:
                best_ratio = r
                best = (s["begin"], s["end"])
    if best and best_ratio >= 0.58:
        return best

    return None


def extract_description_for_record(
    rec: dict,
    page: Optional[fitz.Page],
    full_text: str,
    sentence_index: List[Dict[str, Any]],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    page_idx = rec["page_index"]
    items: List[Tuple[int, str, Optional[Tuple[int, int]]]] = []

    # 1) Caption
    caption = (rec.get("caption_text") or "").strip()
    if caption:
        cap_clean = clean_description_text(caption)
        if cap_clean and len(cap_clean) > 10 and not is_math_like(cap_clean):
            if calculate_semantic_relevance(cap_clean) > 0.1:
                cap_pos = find_sentence_position_best(full_text, sentence_index, page_idx, caption)
                begin = cap_pos[0] if cap_pos else 0
                items.append((begin, _norm(cap_clean), cap_pos))

    # 2) Nearby paragraphs fallback
    if not items and page is not None and rec.get("expanded_bbox"):
        raw_paras = extract_paragraphs_around_figure(
            page, fitz.Rect(*rec["expanded_bbox"]), vertical_margin_pt=120.0
        )
        candidate_sents = select_description_sentences(raw_paras, max_sentences=3)

        for sent in candidate_sents:
            sent_raw = sent.strip()
            if len(sent_raw) < 10:
                continue
            sent_clean = _norm(clean_description_text(sent_raw))
            pos = find_sentence_position_best(full_text, sentence_index, page_idx, sent_raw)
            begin = pos[0] if pos else 0
            items.append((begin, sent_clean, pos))

    items.sort(key=lambda x: x[0])

    seen = set()
    descriptions: List[str] = []
    positions: List[Tuple[int, int]] = []

    for _, txt, pos in items:
        if pos is None:
            continue
        key = txt.lower()
        if key not in seen:
            seen.add(key)
            descriptions.append(txt)
            positions.append(pos)

    return descriptions, positions


# CSV writer (submission)

def write_paper_counts_csv(tracking, exam_id):
    filename = f"paper_list_counts_{exam_id}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arxiv_id", "image_count"])
        for pid, info in tracking.items():
            val = info["image_count"] if info["processed"] else ""
            writer.writerow([pid, val])


# MAIN

def main():
    setup_directories()

    all_paper_ids = load_paper_list(PAPER_LIST_FILE)
    if not all_paper_ids:
        logger.error("No IDs loaded.")
        return

    logger.info("Loading circuit classifier model...")
    model = load_circuit_model()

    total_saved_images = 0          
    total_records_written = 0       

    dataset: Dict[str, Any] = {}
    paper_tracking = {pid: {"processed": False, "image_count": 0} for pid in all_paper_ids}

    agg_stats: Dict[str, int] = {
        "clusters_raw_total": 0,
        "clusters_merged_total": 0,
        "candidate_regions_total": 0,
        "saved_circuit": 0,
        "reject_small_bbox": 0,
        "reject_text_by_content": 0,
        "reject_text_by_geometry": 0,
        "reject_pred_not_circuit": 0,
        "reject_pred_lowprob": 0,
        "reject_pred_very_lowprob": 0,
    }

    for idx, arxiv_id in enumerate(all_paper_ids, start=1):
        if total_records_written >= TARGET_NUM_CIRCUITS:
            break

        logger.info(f"[{idx}/{len(all_paper_ids)}] Processing {arxiv_id}")
        pdf_path = download_pdf(arxiv_id)
        if not pdf_path:
            continue

        try:
            doc_page_numbers = extract_footer_page_numbers(pdf_path)
            full_text, sentence_index = build_full_text(pdf_path)
        except Exception as e:
            logger.error(f"Text prep error {arxiv_id}: {e}")
            continue

        remaining = TARGET_NUM_CIRCUITS - total_records_written
        if remaining <= 0:
            break

        try:
            extraction_result = extract_and_classify_clusters(
                model,
                pdf_path=pdf_path,
                arxiv_id=arxiv_id,
                prob_threshold=0.80,
                doc_page_numbers=doc_page_numbers,
                max_to_save=remaining,
            )

            if isinstance(extraction_result, tuple) and len(extraction_result) == 3:
                num_circuits, records, stats = extraction_result
            else:
                num_circuits, records = extraction_result
                stats = {}

            # Per-PDF stats (printed + aggregated)
            per_pdf = {
                "clusters_raw_total": int(stats.get("clusters_raw_total", 0)),
                "clusters_merged_total": int(stats.get("clusters_merged_total", 0)),
                "candidate_regions_total": int(stats.get("candidate_regions_total", 0)),
                "saved_circuit": int(stats.get("saved_circuit", num_circuits)),
                "reject_small_bbox": int(stats.get("reject_small_bbox", 0)),
                "reject_text_by_content": int(stats.get("reject_text_by_content", 0)),
                "reject_text_by_geometry": int(stats.get("reject_text_by_geometry", 0)),
                "reject_pred_not_circuit": int(stats.get("reject_pred_not_circuit", 0)),
                "reject_pred_lowprob": int(stats.get("reject_pred_lowprob", 0)),
                "reject_pred_very_lowprob": int(stats.get("reject_pred_very_lowprob", 0)),
            }
            for k, v in per_pdf.items():
                agg_stats[k] += int(v)

            logger.info(
                " -> Extraction stats: raw_clusters=%d, merged_clusters=%d, candidates=%d, saved_circuit=%d, "
                "rejected_small=%d, rejected_text_content=%d, rejected_text_geometry=%d, "
                "rejected_not_circuit=%d, rejected_lowprob=%d, rejected_very_lowprob=%d",
                per_pdf["clusters_raw_total"],
                per_pdf["clusters_merged_total"],
                per_pdf["candidate_regions_total"],
                per_pdf["saved_circuit"],
                per_pdf["reject_small_bbox"],
                per_pdf["reject_text_by_content"],
                per_pdf["reject_text_by_geometry"],
                per_pdf["reject_pred_not_circuit"],
                per_pdf["reject_pred_lowprob"],
                per_pdf["reject_pred_very_lowprob"],
            )

            paper_tracking[arxiv_id]["processed"] = True
            paper_tracking[arxiv_id]["image_count"] = num_circuits

            total_saved_images += int(per_pdf["saved_circuit"])

            doc = fitz.open(pdf_path)
            try:
                for rec in records:
                    if total_records_written >= TARGET_NUM_CIRCUITS:
                        break

                    try:
                        fname = rec["image_filename"]
                        image_path_full = rec["image_path"]
                        page_index = rec["page_index"]

                        page = doc[page_index] if page_index < len(doc) else None

                        descriptions, positions = extract_description_for_record(
                            rec, page, full_text, sentence_index
                        )

                        raw_context = ""
                        if page and rec.get("expanded_bbox"):
                            raw_paras = extract_paragraphs_around_figure(
                                page, fitz.Rect(*rec["expanded_bbox"])
                            )
                            raw_context = " ".join([p["text"] for p in raw_paras])

                        extraction = extract_problem_and_gates(
                            caption_text=(rec.get("caption_text") or ""),
                            context_text=raw_context,
                            image_path=image_path_full,
                        )

                        dataset[fname] = {
                            "arxiv_number": rec["arxiv_id"],
                            "page_number": rec["page_number"],
                            "figure_number": normalize_figure_number(rec["figure_number"]),
                            "quantum_gates": extraction["quantum_gates"],
                            "quantum_problem": extraction["quantum_problem"],
                            "descriptions": descriptions,
                            "text_positions": positions,
                        }

                        total_records_written += 1

                    except Exception as e:
                        logger.warning(f" -> metadata build failed for {rec.get('image_filename')}: {e}")
                        # keep images_4 consistent with JSON
                        try:
                            os.remove(rec.get("image_path", ""))
                        except Exception:
                            pass
                        continue

            finally:
                doc.close()

            # Save progress after each PDF
            with open(TARGET_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Processing error {arxiv_id}: {e}")
            continue

    write_paper_counts_csv(paper_tracking, "4")

    logger.info(
        "Extraction totals: raw_clusters=%d, merged_clusters=%d, candidates=%d, saved_circuit=%d, rejected_small=%d, "
        "rejected_text_content=%d, rejected_text_geometry=%d, rejected_not_circuit=%d, "
        "rejected_lowprob=%d, rejected_very_lowprob=%d",
        agg_stats.get("clusters_raw_total", 0),
        agg_stats.get("clusters_merged_total", 0),
        agg_stats.get("candidate_regions_total", 0),
        agg_stats.get("saved_circuit", 0),
        agg_stats.get("reject_small_bbox", 0),
        agg_stats.get("reject_text_by_content", 0),
        agg_stats.get("reject_text_by_geometry", 0),
        agg_stats.get("reject_pred_not_circuit", 0),
        agg_stats.get("reject_pred_lowprob", 0),
        agg_stats.get("reject_pred_very_lowprob", 0),
    )

    logger.info("Totals: saved_images=%d, records_written=%d", total_saved_images, total_records_written)
    logger.info("Done.")


if __name__ == "__main__":
    main()
