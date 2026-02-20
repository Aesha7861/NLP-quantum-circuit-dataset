"""
Circuit extraction and classification from PDF pages.

NLP/ML Techniques Used:
  - Image classification (CNN: circuit vs text vs table)
  - Content-based semantic filtering (gate keyword detection)
  - Geometric layout analysis (block detection + clustering)
  - Heuristic-based text classification (prose vs technical)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any
import fitz  # PyMuPDF
import numpy as np
import cv2
from circuit_classifier import predict_image
from text_layout_utils import detect_figure_number_and_caption
from gate_problem_extractor import extract_problem_and_gates  

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent        
PROJECT_ROOT = BASE_DIR.parent 

IMAGE_OUT_ROOT = PROJECT_ROOT / "images_4"
TMP_ROOT = IMAGE_OUT_ROOT / "tmp"

# EXTRACTION PARAMETERS (TUNABLE)

ZOOM = 3.5
MERGE_HORIZ_GAP = 250.0
MERGE_VERT_GAP = 60.0

# Padding adjustments
PAD_LR_FACTOR = 0.26
PAD_LR_MIN = 70
PAD_LR_MAX = 240
PAD_TOP_FACTOR = 0.035
PAD_TOP_MIN = 3
PAD_TOP_MAX = 12
PAD_BOTTOM_FACTOR = 0.02
PAD_BOTTOM_MIN = 3
PAD_BOTTOM_MAX = 10

# Expansion for multi-scale detection
EXPAND_LR_STEP = 70
EXPAND_TOP_STEP = 10
EXPAND_BOTTOM_STEP = 6
MAX_EXPAND_ITERS = 4
FINAL_MARGIN_LR = 14
FINAL_MARGIN_TOP = 6

# Thresholds
PROB_THRESHOLD_DEFAULT = 0.80
PROB_THRESHOLD_TEXTY = 0.95

# NLP LEXICONS FOR TEXT CLASSIFICATION

# English stopwords: if many present, text is likely prose
ENGLISH_STOPWORDS = {
    "the", "and", "of", "in", "to", "for", "with", "by", "on", "at", "is",
    "are", "this", "that", "we", "our", "their", "which", "from", "an",
    "or", "but", "not", "can", "has", "have", "these", "those", "such",
    "using", "used", "via", "based",
}

# Academic/bibliographic markers: reject if present
ACADEMIC_MARKERS = {
    "doi", "vol", "no", "pp", "phys", "rev", "lett", "jour", "proc", "conf",
    "symp", "trans", "soc", "adv", "mat", "int", "sci", "res", "bull", "ann",
    "isbn", "issn", "editor", "edition", "page", "pages", "submitted",
    "accepted", "published", "arxiv", "preprint", "journal", "letters",
    "communications", "bibliography", "references", "contract", "grant",
    "award", "supported", "funded", "agency", "foundation", "ministry",
    "department", "dept", "university", "institute", "school", "center",
    "laboratory", "college", "corporation", "acknowledgments", "acknowledge",
    "thanks",
}

# Gate keywords: if present, likely a circuit (rescue from rejection)
GATE_KEYWORDS = {
    "h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "u", "u3", "u2", "u1",
    "cx", "cz", "cnot", "swap", "ccx", "toffoli", "|0", "⟩", "⟨", "measure",
    "qft", "control", "target", "qubit", "circuit", "gate",
}

# TEXT CLASSIFICATION FOR CONTENT DETECTION

def is_text_by_content(page: fitz.Page, bbox: fitz.Rect) -> bool:
    """
    Classify region using semantic content (lexicon-based text categorization).
    
    Technique: Lexicon-based information extraction + keyword matching.
    Returns: TRUE if clearly text/prose, FALSE if possibly circuit.
    """
    text_content = page.get_text("text", clip=bbox).lower()
    tokens = [t.strip(".,()[]:;") for t in text_content.split()]

    if not tokens:
        return False

    # Count positive and negative indicators
    english_hits = 0
    academic_hits = 0
    gate_hits = 0

    for t in tokens:
        if t in ENGLISH_STOPWORDS:
            english_hits += 1
        if t in ACADEMIC_MARKERS:
            academic_hits += 1
        if t in GATE_KEYWORDS:
            gate_hits += 1

    # If circuit keywords present, rescue from text classification
    if gate_hits >= 1:
        return False

    # Hard reject bibliography/academic markers
    if academic_hits >= 2:
        return True

    # If mostly English prose, likely text
    if len(tokens) > 6:
        ratio = english_hits / len(tokens)
        if ratio > 0.20:
            return True

    return False


def is_text_by_geometry(bgr: np.ndarray) -> bool:
    """
    Classify region using visual features (geometry + texture heuristics).
    
    Technique: Image-level feature extraction + heuristic rules.
    Returns: TRUE if looks like text block, FALSE if looks like circuit.
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Feature 1: Aspect ratio 
    if H > W * 3.5:
        return True

    # Feature 2: Pixel density
    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    density = cv2.countNonZero(bw) / (H * W)
    if density > 0.18:
        return True

    # Feature 3: Gate/box detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    has_gate_box = False
    for c in contours:
        area = cv2.contourArea(c)
        if area > (H * W) * 0.02:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h
            if 0.3 < aspect < 3.0:
                has_gate_box = True
                break

    if has_gate_box:
        return False

    # Feature 4: Wire detection (horizontal lines)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=int(W * 0.25),
        maxLineGap=15,
    )

    has_wire = False
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 3 or angle > 177:
                has_wire = True
                break

    if has_wire:
        return False

    return True


# GEOMETRIC PROCESSING

def merge_nearby_rects(
    rects: List[fitz.Rect], horiz_gap: float, vert_gap: float
) -> List[fitz.Rect]:
    """
    Merge nearby rectangles (clustering-like operation).
    Technique: Hierarchical geometric merging.
    """
    if not rects:
        return []

    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged: List[fitz.Rect] = []

    for rect in rects:
        merged_any = False

        for i, m in enumerate(merged):
            # Check intersection
            if rect.intersects(m):
                merged[i] = m | rect
                merged_any = True
                break

            # Check horizontal proximity
            h_dist = min(abs(rect.x0 - m.x1), abs(m.x0 - rect.x1))
            v_dist = min(abs(rect.y0 - m.y1), abs(m.y0 - rect.y1))
            y_overlap = min(rect.y1, m.y1) - max(rect.y0, m.y0)
            x_overlap = min(rect.x1, m.x1) - max(rect.x0, m.x0)

            if h_dist < horiz_gap and y_overlap > 0:
                merged[i] = m | rect
                merged_any = True
                break

            if v_dist < vert_gap and x_overlap > 0:
                merged[i] = m | rect
                merged_any = True
                break

        if not merged_any:
            merged.append(rect)

    return merged


def pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    """Convert PyMuPDF pixmap to OpenCV BGR format."""
    n = pix.n
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, n)
    if n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def trim_bbox_using_pdf_text(page: fitz.Page, bbox: fitz.Rect) -> fitz.Rect:
    """
    Trim bbox by removing academic markers above/below.
    Technique: Layout-aware bbox refinement using semantic content.
    """
    d = page.get_text("dict", clip=bbox)
    new_y0, new_y1 = bbox.y0, bbox.y1

    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue

        bx0, _, bx1, by1 = b.get("bbox")

        if (bx1 - bx0) > bbox.width * 0.9:
            text = "".join(
                s.get("text", "")
                for l in b.get("lines", [])
                for s in l.get("spans", [])
            ).lower()

            if any(m in text for m in ACADEMIC_MARKERS) or "figure" in text:
                if by1 < bbox.y0 + (bbox.height * 0.25):
                    new_y0 = max(new_y0, by1)
                if b.get("bbox")[1] > bbox.y1 - (bbox.height * 0.25):
                    new_y1 = min(new_y1, b.get("bbox")[1])

    if new_y1 - new_y0 < 20:
        return bbox

    return fitz.Rect(bbox.x0, new_y0, bbox.x1, new_y1)


# MAIN EXTRACTION FUNCTION

def extract_and_classify_clusters(
    model,
    pdf_path: str,
    arxiv_id: str,
    prob_threshold: float = PROB_THRESHOLD_DEFAULT,
    zoom: float = ZOOM,
    merge_horiz_gap: float = MERGE_HORIZ_GAP,
    merge_vert_gap: float = MERGE_VERT_GAP,
    doc_page_numbers: Optional[list] = None,
    on_save: Optional[Callable[..., Any]] = None,
    max_to_save: Optional[int] = None,
    **_ignored_kwargs,
) -> Tuple[int, list, dict]:

    IMAGE_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom_mat = fitz.Matrix(zoom, zoom)

    total_saved = 0
    records = []
    stop_early = False 

    stats = {
        "pages_total": 0,
        "pages_with_drawings": 0,
        "clusters_raw_total": 0,
        "clusters_merged_total": 0,
        "candidate_regions_total": 0,
        "reject_small_bbox": 0,
        "reject_text_by_content": 0,
        "reject_text_by_geometry": 0,
        "reject_pred_not_circuit": 0,
        "reject_pred_lowprob": 0,
        "reject_pred_very_lowprob": 0,
        "saved_circuit": 0,
    }

    safe_id = arxiv_id.replace("/", "_").replace(":", "_")

    try:
        for page_index, page in enumerate(doc):
            stats["pages_total"] += 1

            if not page.get_drawings():
                continue

            stats["pages_with_drawings"] += 1

            try:
                clusters = page.cluster_drawings()
            except Exception:
                continue

            stats["clusters_raw_total"] += len(clusters) if clusters else 0

            merged = merge_nearby_rects(clusters, merge_horiz_gap, merge_vert_gap)
            stats["clusters_merged_total"] += len(merged) if merged else 0

            for cluster_idx, bbox in enumerate(merged, start=1):
                stats["candidate_regions_total"] += 1

                if bbox.width < 50 or bbox.height < 30:
                    stats["reject_small_bbox"] += 1
                    continue

                if is_text_by_content(page, bbox):
                    stats["reject_text_by_content"] += 1
                    continue

                pad_lr = min(PAD_LR_MAX, max(PAD_LR_MIN, int(PAD_LR_FACTOR * bbox.width)))
                pad_tb = min(PAD_TOP_MAX, max(PAD_TOP_MIN, int(PAD_TOP_FACTOR * bbox.height)))

                expanded_bbox = fitz.Rect(
                    bbox.x0 - pad_lr,
                    bbox.y0 - pad_tb,
                    bbox.x1 + pad_lr,
                    bbox.y1 + pad_tb,
                )
                expanded_bbox = expanded_bbox & page.rect
                expanded_bbox = trim_bbox_using_pdf_text(page, expanded_bbox)

                pix = page.get_pixmap(matrix=zoom_mat, clip=expanded_bbox, alpha=False)
                bgr = pixmap_to_bgr(pix)

                if is_text_by_geometry(bgr):
                    stats["reject_text_by_geometry"] += 1
                    continue

                tmp = TMP_ROOT / f"{safe_id}_p{page_index+1}_c{cluster_idx}.png"
                try:
                    pix.save(tmp.as_posix())

                    label, prob = predict_image(model, tmp.as_posix())
                    thresh = prob_threshold

                    # mutually exclusive reject buckets
                    if prob <= 0.70:
                        stats["reject_pred_very_lowprob"] += 1
                        continue
                    if label != "circuit":
                        stats["reject_pred_not_circuit"] += 1
                        continue
                    if prob < thresh:
                        stats["reject_pred_lowprob"] += 1
                        continue

                    out_name = f"{safe_id}_page{page_index+1}_clusters{cluster_idx}.png"
                    out_path = (IMAGE_OUT_ROOT / out_name).as_posix()
                    pix.save(out_path)

                    total_saved += 1
                    stats["saved_circuit"] += 1  

                    # Determine page number
                    if (
                        doc_page_numbers is not None
                        and page_index < len(doc_page_numbers)
                        and doc_page_numbers[page_index] is not None
                    ):
                        page_number = doc_page_numbers[page_index]
                    else:
                        page_number = page_index + 1

                    fig_number, caption_text = detect_figure_number_and_caption(page, expanded_bbox)

                    record = {
                        "image_filename": out_name,
                        "image_path": out_path,
                        "arxiv_id": arxiv_id,
                        "page_index": page_index,
                        "page_number": page_number,
                        "cluster_idx": cluster_idx,
                        "expanded_bbox": [
                            float(expanded_bbox.x0),
                            float(expanded_bbox.y0),
                            float(expanded_bbox.x1),
                            float(expanded_bbox.y1),
                        ],
                        "figure_number": fig_number,
                        "caption_text": caption_text,
                    }
                    records.append(record)

                    if on_save is not None:
                        try:
                            on_save(record=record, page=page, expanded_bbox=expanded_bbox)
                        except Exception as e:
                            logger.warning(f"Annotation callback failed for {out_name}: {e}")

                    # correct early stopping
                    if max_to_save is not None and total_saved >= max_to_save:
                        stop_early = True
                        break

                finally:
                    # always try to delete temp
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

            if stop_early:
                break

    finally:
        doc.close()

    return total_saved, records, stats
