"""
Extract numeric AND alphabetic figure numbers from captions.
- Searches captions FIRST (most reliable)
- Extracts: numeric (3), alphabetic (a, b, c), combined (3a, 3.1a)
- Falls back to above/below search if no caption found
- Handles nested formats: Fig 3(a), Fig 3.1.b, etc.
"""

import re
from typing import List, Optional, Tuple, Dict, Any, Union
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

# CAPTION PATTERNS - Handle all figure number formats

_caption_patterns = [
    # Standard: "Figure 3: ..." or "Fig. 3: ..."
    re.compile(r"^\s*fig(?:ure)?\s*\.?\s*([0-9]+)\s*[:\-.\s](.*)$", re.IGNORECASE),
    
    # With alphabetic suffix: "Figure 3a: ...", "Fig. 3(a): ...", "Fig. 3.a: ..."
    re.compile(r"^\s*fig(?:ure)?\s*\.?\s*([0-9]+)\s*[.\-()]*\s*([a-zA-Z])\s*[:\-.\s](.*)$", re.IGNORECASE),
    
    # With subpart: "Figure 3.1: ...", "Fig. 3.1.a: ..."
    re.compile(r"^\s*fig(?:ure)?\s*\.?\s*([0-9]+\.[0-9a-zA-Z.]+)\s*[:\-.\s](.*)$", re.IGNORECASE),
    
    # JUST ALPHABETIC: "Figure a: ...", "Fig. b: ..."
    re.compile(r"^\s*fig(?:ure)?\s*\.?\s*([a-zA-Z])\s*[:\-.\s](.*)$", re.IGNORECASE),
    
    # JUST ALPHABETIC with capitals: "Figure A: ...", "Fig. B: ..."
    re.compile(r"^\s*fig(?:ure)?\s*\.?\s*([A-Z])\s*[:\-.\s](.*)$", re.IGNORECASE),
    
    # Parenthetical format: "(a) Circuit description...", "(b) Implementation..."
    re.compile(r"^\s*\(([a-zA-Z0-9]+)\)\s*[:\-.\s](.*)$", re.IGNORECASE),
    
    # Dash format: "a - Circuit...", "b - Description..."
    re.compile(r"^\s*([a-zA-Z0-9]+)\s*[-–—]\s*(.*)$", re.IGNORECASE),
]

# FIGURE NUMBER EXTRACTION (Handles all formats)

def extract_figure_number_from_text(text: str) -> Optional[Union[str, int]]:
    """
    Extract figure number in ANY format from text.
    
    Returns: 
      - int: if numeric (e.g., 3, 7)
      - str: if alphabetic (e.g., 'a', 'b', 'Fig3a', '3.1')
      - None: if not found
    
    Technique: Multi-pattern regex matching (rule-based IE).
    """
    t = text.strip()
    if not t:
        return None
    
    # Try each pattern
    for pattern in _caption_patterns:
        m = pattern.match(t)
        if m:
            try:
                # Group 1 is the figure number
                fig_num_str = m.group(1).strip()
                if not fig_num_str:
                    continue
                
                # Try to parse as int first (3, 7, 15)
                try:
                    return int(fig_num_str)
                except ValueError:
                    # Not purely numeric, keep as string (a, b, 3a, 3.1, etc.)
                    return fig_num_str
                    
            except (ValueError, IndexError):
                continue
    
    return None


def extract_caption_from_text(text: str) -> Optional[str]:
    """
    Extract caption text (description) from a line containing figure label.
    
    Returns: caption text, or None if no caption found
    """
    t = text.strip()
    if not t:
        return None
    
    for pattern in _caption_patterns:
        m = pattern.match(t)
        if m:
            try:
                # Try to get last group (usually the caption)
                groups = m.groups()
                if len(groups) >= 2:
                    # For most patterns, caption is in last group
                    caption = groups[-1].strip()
                    if caption and len(caption) > 3:
                        return caption
                
                # Fallback: return entire remaining text
                if len(groups) >= 1:
                    return t[m.end(1):].strip(r": \-")
                    
            except (IndexError, AttributeError):
                continue
    
    return None


# CAPTION-FIRST DETECTION (Search in captions first, then above/below)

def detect_figure_number_and_caption(
    page: fitz.Page,
    figure_bbox: fitz.Rect,
    vertical_margin: float = 150.0,
    search_above: bool = True,
    search_below: bool = True,
) -> Tuple[Optional[Union[str, int]], Optional[str]]:
    """
    Extract figure number (numeric or alphabetic) + caption.
    
    Strategy:
      1. Search BELOW figure first (most common: caption below diagram)
      2. Then search ABOVE figure (alternative placement)
      3. For each line, extract figure number using all patterns
      4. Return first match with valid figure number
    
    Returns: (figure_number, caption_text)
      - figure_number can be: int (3), str ('a', '3a', '3.1'), or None
      - caption_text can be: str (caption) or None
    
    Technique: Multi-pattern IE + layout-aware geometric search.
    """
    data = page.get_text("dict")
    fig_x0, fig_y0, fig_x1, fig_y1 = figure_bbox
    
    candidates: List[Tuple[float, Union[str, int], str]] = [] 
    
    # STRATEGY: Search BELOW first (most likely), then ABOVE
    search_order = []
    if search_below:
        search_order.append(("below", fig_y1, min(page.rect.y1, fig_y1 + vertical_margin)))
    if search_above:
        search_order.append(("above", max(0, fig_y0 - vertical_margin), fig_y0))
    
    # Search in defined order (below first, then above)
    for region_name, search_y0, search_y1 in search_order:
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # Only text blocks
                continue
            
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                
                y0 = min(s["bbox"][1] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)
                
                # Check if line is within search region
                if y0 < search_y0 or y0 > search_y1:
                    continue
                
                line_text = "".join(s.get("text", "") for s in spans).strip()
                if not line_text or len(line_text) < 2:
                    continue
                
                # Try to extract figure number from this line
                fig_num = extract_figure_number_from_text(line_text)
                
                if fig_num is not None:
                    # Distance metric: prefer closer lines
                    if region_name == "below":
                        distance = y0 - fig_y1
                    else:
                        distance = fig_y0 - y1
                    
                    candidates.append((distance, fig_num, line_text))
    
    # If no candidates found, return None
    if not candidates:
        logger.debug(f"No figure number found near bbox {figure_bbox}")
        return None, None
    
    # Sort by distance (closest first)
    candidates.sort(key=lambda t: t[0])
    
    # Take closest match
    distance, fig_num, best_line = candidates[0]
    
    # Extract caption from the line containing the figure number
    caption = extract_caption_from_text(best_line)
    
    logger.debug(f"Found figure: num={fig_num}, distance={distance:.1f}pt, caption={caption[:50] if caption else 'N/A'}")
    
    return fig_num, caption if caption else best_line


# NLP CONSTANTS (Keep existing ones)

MATH_CHARS = set("=+-*/<>∑∏√∞≈≠≤≥()[]{}^_|\\∀∃∈∉⊂⊆⊕⊗∧∨¬⇒⇔")
WEIRD_CHARS = set("")

CIRCUIT_KEYWORDS = {
    "circuit", "gate", "qubit", "oracle", "ancilla", "phase",
    "grover", "qft", "qram", "measurement", "control", "target",
    "superposition", "entanglement", "unitary", "operation",
    "implementation", "cnot", "cx", "swap", "hadamard", "pauli",
    "rotation", "eigen", "eigenstate", "amplitude", "quantum"
}

BIBLIOGRAPHY_KEYWORDS = {
    "doi", "vol", "issue", "pages", "arxiv", "preprint", "references",
    "bibliography", "acknowledgments", "funding", "grant", "contract",
    "supported by", "approved by", "editor", "edition", "journal",
    "publisher", "isbn", "issn", "submitted", "accepted", "published"
}

ENGLISH_STOPWORDS = {
    "the", "and", "of", "in", "to", "for", "with", "by", "on", "at", "is",
    "are", "this", "that", "we", "our", "their", "which", "from", "an",
    "or", "but", "not", "can", "has", "have", "these", "those", "such",
    "using", "used", "via", "based", "where", "when", "what", "how", "if"
}

# EXISTING HELPER FUNCTIONS (Keep all previous functions)

FOOTER_HEIGHT_PT = 40
_footer_page_pattern = re.compile(r"(?:page\s*)?(\d{1,4})(?:\s*/\s*\d{1,4})?$", re.IGNORECASE)
_sentence_split_pattern = re.compile(r"([.!?])")


def _extract_page_number_from_footer_text(text: str) -> Optional[int]:
    """Extract a page number from footer text using regex pattern matching."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        if line.isdigit():
            return int(line)
        m = _footer_page_pattern.search(line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def extract_footer_page_numbers(pdf_path: str) -> List[Optional[int]]:
    """For each page, extract document page number from footer."""
    doc = fitz.open(pdf_path)
    result: List[Optional[int]] = []
    for page in doc:
        rect = page.rect
        footer_band = fitz.Rect(rect.x0, rect.y1 - FOOTER_HEIGHT_PT, rect.x1, rect.y1)
        footer_text = page.get_text("text", clip=footer_band)
        page_num = _extract_page_number_from_footer_text(footer_text)
        result.append(page_num)
    doc.close()
    return result


def build_full_text(pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Build linearized full_text for the entire PDF + sentence index."""
    doc = fitz.open(pdf_path)
    full_text_parts: List[str] = []
    sentence_index: List[Dict[str, Any]] = []
    current_pos = 0

    for page_index, page in enumerate(doc):
        page_text = page.get_text("text")
        if not page_text:
            continue

        chunks = _sentence_split_pattern.split(page_text)
        if not chunks:
            continue

        buf = ""
        for chunk in chunks:
            if not chunk:
                continue
            if _sentence_split_pattern.fullmatch(chunk):
                buf += chunk
                sent = buf.strip()
                if sent:
                    begin = current_pos
                    full_text_parts.append(sent)
                    current_pos += len(sent)
                    sentence_index.append({
                        "page_index": page_index,
                        "text": sent,
                        "begin": begin,
                        "end": current_pos,
                    })
                buf = ""
            else:
                buf = buf + chunk if buf else chunk

        sent = buf.strip()
        if sent:
            begin = current_pos
            full_text_parts.append(sent)
            current_pos += len(sent)
            sentence_index.append({
                "page_index": page_index,
                "text": sent,
                "begin": begin,
                "end": current_pos,
            })

        sep = "\n"
        full_text_parts.append(sep)
        current_pos += len(sep)

    doc.close()
    full_text = "".join(full_text_parts)
    return full_text, sentence_index


def is_math_like(text: str) -> bool:
    """Character-level text classification: detect if text is mostly math/equation."""
    t = text.strip()
    if not t:
        return False

    if re.match(r"^\s*fig(?:ure)?\.\s*\d+", t, re.IGNORECASE):
        return True

    math_count = sum(1 for ch in t if ch in MATH_CHARS or ch in WEIRD_CHARS)
    digit_count = sum(1 for ch in t if ch.isdigit())
    letter_count = sum(1 for ch in t if ch.isalpha())
    total = max(len(t), 1)

    if math_count / total > 0.10:
        return True
    if digit_count / total > 0.40:
        return True
    if letter_count / total < 0.20:
        return True
    if len(t) < 5:
        return True

    return False


def clean_description_text(text: str) -> str:
    """Preprocessing: remove figure labels, collapse whitespace, strip glyphs."""
    t = text.strip()

    # Strip only the leading figure prefix, keep the caption content.
    t = re.sub(
        r"^\s*(?:figure|fig)\.?\s*\d+(?:\.\d+)*\s*(?:\([a-z0-9]+\))?\s*[:\-]?\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"^\s*fig(?:ure)?\.\s*\d+[a-zA-Z()]*\s*[:\-.\s]+",
        "",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(
        r"Figure\s*\d+[a-zA-Z()]*\s*[:\-]?\s*.*$",
        "",
        t,
        flags=re.IGNORECASE,
    )

    t = re.sub(r"\s+", " ", t)
    t = "".join(ch for ch in t if ch not in WEIRD_CHARS)

    return t.strip()


def calculate_semantic_relevance(text: str) -> float:
    """Score text for relevance to quantum circuits (0.0 to 1.0)."""
    t = text.lower()
    tokens = re.findall(r"\w+", t)
    if not tokens:
        return 0.0

    circuit_hits = sum(1 for tok in tokens if tok in CIRCUIT_KEYWORDS)
    bib_hits = sum(1 for tok in tokens if tok in BIBLIOGRAPHY_KEYWORDS)

    if bib_hits > 0:
        return 0.0

    max_hits = max(len(CIRCUIT_KEYWORDS), 1)
    score = min(circuit_hits / max_hits, 1.0)
    return score


def is_prose_like(text: str) -> bool:
    """Detect if text is natural language prose (not table/list)."""
    t = text.lower()
    tokens = [tok.strip(".,()[];:") for tok in t.split()]
    if not tokens:
        return False

    english_hits = sum(1 for tok in tokens if tok in ENGLISH_STOPWORDS)
    ratio = english_hits / len(tokens) if tokens else 0

    return ratio > 0.15


def extract_paragraphs_around_figure(
    page: fitz.Page,
    figure_bbox: fitz.Rect,
    vertical_margin_pt: float = 120.0,
) -> List[Dict[str, Any]]:
    """Extract text blocks (paragraphs) near figure using layout geometry."""
    fig_x0, fig_y0, fig_x1, fig_y1 = figure_bbox
    
    search_y0 = max(0, fig_y0 - vertical_margin_pt)
    search_y1 = min(page.rect.y1, fig_y1 + vertical_margin_pt)

    data = page.get_text("dict")
    paragraphs: List[Dict[str, Any]] = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_y0 = min(line["bbox"][1] for line in block.get("lines", []))
        block_y1 = max(line["bbox"][3] for line in block.get("lines", []))

        if block_y1 < search_y0 or block_y0 > search_y1:
            continue

        para_text = ""
        para_sentences = []
        
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if line_text:
                para_text += " " + line_text
                para_sentences.append({
                    "text": line_text,
                    "y0": line["bbox"][1],
                    "y1": line["bbox"][3],
                })

        para_text = para_text.strip()
        if not para_text:
            continue

        paragraphs.append({
            "text": para_text,
            "y0": block_y0,
            "y1": block_y1,
            "sentences": para_sentences,
        })

    return paragraphs


def select_description_sentences(
    paragraphs: List[Dict[str, Any]],
    max_sentences: int = 3,
) -> List[str]:
    """From list of paragraphs, select best sentences for description."""
    scored_sentences: List[Tuple[float, str]] = []

    for para in paragraphs:
        para_text = para["text"]
        
        if is_math_like(para_text):
            continue

        sent_matches = re.split(r"[.!?]+", para_text)
        
        for sent in sent_matches:
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue

            if re.match(r"^\s*fig(?:ure)?\.?\s*\d+", sent, re.IGNORECASE):
                continue

            if is_math_like(sent):
                continue

            relevance = calculate_semantic_relevance(sent)
            prose = 1.0 if is_prose_like(sent) else 0.5
            score = relevance * 0.6 + prose * 0.4

            if score > 0:
                scored_sentences.append((score, sent))

    scored_sentences.sort(key=lambda x: -x[0])
    selected = [sent for _, sent in scored_sentences[:max_sentences]]
    return selected
