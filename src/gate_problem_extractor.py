"""
Quantum Gates and Problem Type Extraction (Strict & Image-Aware)

Features:
1. Image-First: Uses OCR (Tesseract) to find gates inside the circuit diagram.
2. Strict Text: Distinguishes variables (x, y, z) from Gates (X gate, Pauli-Z) using context.
3. Hierarchical Classification: Identifies specific algorithms over generic ones.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Any

# Try importing OCR libraries 
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = logging.getLogger(__name__)

# 1. Gate Lexicon

# Gates that are unique enough to match without context (e.g. "CNOT")
MULTI_CHAR_GATES = {
    "hadamard": "Hadamard",
    "cnot": "CNOT",
    "cx": "CNOT",
    "cz": "CZ",
    "ccx": "Toffoli",
    "toffoli": "Toffoli",
    "swap": "SWAP",
    "iswap": "iSWAP",
    "measure": "Measurement",
    "measurement": "Measurement",
    "meter": "Measurement",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "qft": "QFT",
    "iqft": "iQFT",
    "reset": "Reset",
    "barrier": "Barrier",
    "oracle": "Oracle",
}

# Single-letter gates that require STRICT context in text (e.g. "X gate" vs variable "x")
SINGLE_CHAR_GATES = {
    "h": "Hadamard",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "u": "U",
}

# Context words that indicate a single letter is actually a gate in text
PRE_CONTEXT_KEYWORDS = {"pauli", "gate", "rotation", "controlled", "ctrl", "phase", "single", "type"}
POST_CONTEXT_KEYWORDS = {"gate", "rotation", "operation", "operator", "axis", "basis", "error", "pulse"}

# Words to explicitly ignore even if they look like gates
NON_GATES = {
    "grover", "algorithm", "circuit", "scheme", "method", "step", "processor",
    "figure", "fig", "eq", "equation", "table", "section"
}

# 2. Image OCR Extraction (First Priority)

def extract_gates_from_image(image_path: str) -> List[str]:
    """
    Scans the circuit image for text using Tesseract OCR.
    Prioritizes text found inside the diagram (H, X, QFT, etc).
    """
    if not HAS_OCR or not image_path:
        return []

    found_gates = set()
    try:
        # Load image
        img = Image.open(image_path).convert("L")  # Grayscale
        # Simple binarization to clean up noise
        img = img.point(lambda p: 255 if p > 200 else 0)
        
        # Run OCR
        # --psm 6 assumes a single uniform block of text (good for diagrams)
        # --psm 11 is for sparse text. 6 is usually balanced for diagrams.
        text = pytesseract.image_to_string(img, config="--psm 6")
        
        # Simple tokenization keeping case (important for 'H' vs 'h')
        tokens = re.findall(r"[a-zA-Z0-9]+", text)
        
        for token in tokens:
            lower_token = token.lower()
            
            # 1. Multi-char match (Robust)
            if lower_token in MULTI_CHAR_GATES:
                found_gates.add(MULTI_CHAR_GATES[lower_token])
            
            # 2. Single-char match (Heuristic: Diagram labels are usually Capitalized)
            # If see a capital "H" or "X" in a box, it's a gate.
            elif lower_token in SINGLE_CHAR_GATES:
                if token.isupper():  # Only accept 'H', 'X', 'Z', not 'x', 'z'
                    found_gates.add(SINGLE_CHAR_GATES[lower_token])

    except Exception as e:
        # Don't crash if OCR fails, just log warning
        logger.warning(f"OCR warning for {image_path}: {e}")

    return list(found_gates)


# 3. Strict Text Extraction

def extract_gates_from_text(text: str) -> List[str]:
    """
    Extracts quantum gates from text using STRICT Context analysis.
    Prevents "variable x" from being detected as "X Gate".
    """
    if not text:
        return []

    # Clean text: remove hyphens to join "Pauli-X" -> "Pauli X"
    clean_text = re.sub(r"[-‐‑]", " ", text.lower())
    clean_text = re.sub(r"[^a-z0-9\s]", "", clean_text)
    
    tokens = clean_text.split()
    gates_found = set()

    # A. Scan for Multi-Char Gates
    for token in tokens:
        if token in MULTI_CHAR_GATES and token not in NON_GATES:
            gates_found.add(MULTI_CHAR_GATES[token])

    # B. Scan for Single-Char Gates (Strict Context)
    for i, token in enumerate(tokens):
        if token in SINGLE_CHAR_GATES:
            is_gate = False
            
            # Check Preceding Word
            if i > 0:
                prev = tokens[i-1]
                if prev in PRE_CONTEXT_KEYWORDS:
                    is_gate = True
            
            # Check Following Word
            if i < len(tokens) - 1:
                next_w = tokens[i+1]
                if next_w in POST_CONTEXT_KEYWORDS:
                    is_gate = True

            # Special case: "C-U" implies Controlled-U
            if token == 'u' and ('controlled' in clean_text or 'conditional' in clean_text):
                 is_gate = True

            if is_gate:
                gates_found.add(SINGLE_CHAR_GATES[token])

    return sorted(list(gates_found))


# 4. Hierarchical Problem Classification

def classify_problem(text: str) -> str:
    """
    Classifies the quantum problem. Checks specific algorithms first.
    """
    t = text.lower()

    # Tier 1: Specific Named Algorithms
    if "grover" in t or ("search" in t and "amplitude amplification" in t):
        return "Grover's Algorithm"
    
    if "shor" in t or "factorization" in t or "period finding" in t:
        return "Shor's Algorithm"
    
    if "vqe" in t or "variational quantum eigensolver" in t or "ansatz" in t:
        return "VQE - Variational Quantum Eigensolver"

    if "qaoa" in t or "maxcut" in t:
        return "QAOA - Quantum Optimization"
        
    if "deutsch" in t and "jozsa" in t:
        return "Deutsch-Jozsa Algorithm"

    # Tier 2: Specific Functions
    if "qft" in t or "fourier" in t:
        return "Quantum Fourier Transform"
    
    if "adder" in t or "addition" in t or "multiplier" in t or "arithmetic" in t:
        return "Arithmetic Operation"

    if "phase estimation" in t or "qpe" in t:
        return "Quantum Phase Estimation"

    # Tier 3: General Categories
    if "magic state" in t or "state injection" in t:
        return "Magic State Injection"
    
    # Catch-all for Oracle/Controlled operations
    if any(k in t for k in ["oracle", "conditional gate", "controlled gate", "cφ", "c^k", "black box"]):
        return "Oracle Gate Implementation"
    
    if "state preparation" in t or "initialization" in t:
        return "Quantum State Preparation"
    
    if "simulation" in t and "circuit" in t:
        return "Quantum Circuit Simulation"

    return "Unknown"


# GATE EXTRACTION (AUTHORITY HIERARCHY — CRITICAL)

def extract_problem_and_gates(
    caption_text: str,
    context_text: str,
    image_path: str | None = None,
    page_title: str = "",
) -> Dict[str, List[str] | str]:
    """Extract gates and the most likely "quantum problem" label.

    Gate extraction priority:
      1) OCR the circuit image (if available). If OCR finds at least 1 gate, OCR wins.
      2) Otherwise fall back to strict text extraction from caption + nearby context.

    Problem classification uses caption + context text (not OCR) because algorithm
    names typically live in prose, not inside gate boxes.
    """
    
    img_gates: List[str] = []
    if image_path:
        img_gates = extract_gates_from_image(image_path)

    text_gates: List[str] = []
    combined_text = " ".join(
        t for t in [caption_text, context_text, page_title] if t
    )
    if combined_text:
        text_gates = extract_gates_from_text(combined_text)

    if len(img_gates) >= 1:
        final_gates = sorted(set(img_gates))      # OCR WINS
    else:
        final_gates = sorted(set(text_gates))     # TEXT ONLY IF OCR EMPTY

    
    # ALGORITHM / PROBLEM (PERMISSION GATE)

    if (caption_text and caption_text.strip()) or (context_text and context_text.strip()):
        quantum_problem = classify_problem(
            f"{caption_text} {context_text} {page_title}"
        )
    else:
        quantum_problem = "Unknown"

    return {
        "quantum_gates": final_gates,
        "quantum_problem": quantum_problem,
    }
