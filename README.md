Project: Image-to-Text Dataset for Quantum Computing (Quantum Circuit Figures)
Author: Aesha Gadhiya
Exam ID: 4
Date: 2025-12-22

This repository contains a fully automated, reproducible pipeline that:
1) downloads arXiv PDFs from a fixed ordered list,
2) extracts candidate figure regions from vector drawing primitives in PDFs,
3) filters and classifies those candidates to keep *only quantum circuit diagrams*,
4) extracts structured metadata (page, figure id, gates, problem label, descriptions, text positions),
5) writes a JSON dataset and saves all circuit images as PNG.

The pipeline stops deterministically once it has produced 250 valid circuit images.

----------------------------------------------------------------------
1) Work files
----------------------------------------------------------------------

Outputs created by the main pipeline:
- images_4/                           # saved PNG circuit images
- dataset_4.json                      # JSON dataset (key = image filename)
- paper_list_counts_4.csv             # arxiv_id + number of extracted images per processed paper

Main entry point:
- build_dataset.py

Supporting modules:
- extract_circuit_images.py           # figure candidate extraction + filtering + CNN classification
- text_layout_utils.py                # figure number/caption detection + text extraction + footer page numbers
- gate_problem_extractor.py           # OCR-then-text gate extraction + problem/algorithm classification
- circuit_classifier.py               # inference-only ResNet18 classifier wrapper
- train_circuit_classifier.py         # training script for the CNN

These scripts and the report are the authoritative description of the method:
- build_dataset.py 
- extract_circuit_images.py 
- text_layout_utils.py
- gate_problem_extractor.py 
- circuit_classifier.py 
- train_circuit_classifier.py 
- Project report: NLP_W25_4.pdf 

----------------------------------------------------------------------
2) Environment & prerequisites
----------------------------------------------------------------------

This pipeline is designed for the DC1.07 GPU lab machines (Linux) but also runs on CPU.

Required system tools (recommended):
- Tesseract OCR (for gate extraction inside circuit images)
  * If not installed, OCR is automatically disabled and the code falls back to text-only extraction
    (see HAS_OCR logic in gate_problem_extractor.py).
Python version:
- Python 3.10+ recommended

Core Python packages used:
- PyMuPDF (fitz): PDF parsing, drawings, clustering, pixmap rendering
- OpenCV (cv2): geometry-based text detection in candidate crops
- NumPy
- PyTorch + Torchvision: ResNet18 classifier
- Pillow (PIL)
- arxiv: arXiv API download client
- pytesseract (optional, if OCR enabled)

(Exact versions belong in requirements.txt; see separate requirements work if needed.)

----------------------------------------------------------------------
3) Quick start (dataset creation)
----------------------------------------------------------------------

i) Put the ordered paper list in the same folder as build_dataset.py:
   - paper_list_4.txt

ii) Ensure the trained classifier weights exist in PROJECT_ROOT:
   - resnet18_circuit_classifier.pth
   This path is the default in circuit_classifier.py. 

iii) Run the main pipeline:
   $ python build_dataset.py

What happens:
- PDFs are downloaded to: papers_4/
- Circuit images are saved to: images_4/
- JSON is written incrementally to: dataset_4.json
- A submission CSV is written at the end: paper_list_counts_4.csv

Stopping condition:
- build_dataset.py stops when records_written reaches TARGET_NUM_CIRCUITS = 250. 

----------------------------------------------------------------------
4) Full pipeline explanation (end-to-end)
----------------------------------------------------------------------

The pipeline is intentionally “layered”: cheap deterministic filters first, heavier ML only later.
That is not aesthetic. It is survival strategy against PDF chaos.

4.1 Stage A — Paper list + arXiv PDF download
-----------------------------------------------------------------

File: build_dataset.py 

Key functions:
- load_paper_list(paper_list_4.txt):
  Loads arXiv IDs exactly in the given order and processes them sequentially.
- download_pdf(arxiv_id):
  Uses the arxiv Python library to download each paper PDF to papers_4/.
  If a PDF already exists, it is reused (reproducibility + speed). 

Reproducibility rule:
- The list order is respected and processing is strictly sequential, as required by the assignment. 

4.2 Stage B — Text preparation for traceability
-----------------------------------------------------------------

File: text_layout_utils.py 
Called from: build_dataset.py 

Two important “text infrastructure” outputs are built *before* image extraction metadata is written:
1) doc_page_numbers = extract_footer_page_numbers(pdf_path)
   - Clips a fixed footer band and extracts printed page numbers via regex.
   - If extraction fails for a page, later code falls back to page_index+1. 
2) full_text, sentence_index = build_full_text(pdf_path)
   - Creates a linearized full document text string (full_text).
   - Builds sentence_index entries with:
     {page_index, text, begin, end}
     where (begin,end) are character offsets inside full_text. 

Why this exists:
- The dataset requires text_positions as (begin,end) offsets. 
- Those offsets must be consistent and reproducible. This is why full_text is constructed deterministically.

Meaning of text_positions (IMPORTANT):
- Each tuple (begin, end) is a character-span in full_text returned by build_full_text().
- The description text in descriptions[i] should correspond (approximately) to full_text[begin:end].
- The offsets are *global within the document*, not page-local. 

4.3 Stage C — Candidate figure region extraction from PDF drawings
-----------------------------------------------------------------

File: extract_circuit_images.py 

Quantum circuit figures in arXiv PDFs are often vector drawings, not raster images.
So extraction cannot rely on “embedded images”.

Approach:
- Iterate pages.
- Only process pages with drawings: page.get_drawings()
- Cluster drawing primitives into bounding boxes: page.cluster_drawings()
- Merge nearby/overlapping cluster boxes to form coherent candidate regions:
  merge_nearby_rects(rects, MERGE_HORIZ_GAP, MERGE_VERT_GAP). 

Why merging is necessary:
- Circuit elements (wires, gate boxes) can be fragmented into multiple PDF drawing clusters.
- If you don’t merge, you get split circuits or incomplete crops.

Output of this stage:
- A set of candidate bounding boxes per page, each considered a potential figure region.

4.4 Stage D — Deterministic pre-filters (reject obvious non-figures)
-------------------------------------------------------------------

File: extract_circuit_images.py 

Before applying the CNN, candidates pass strict, deterministic filters:

(1) Size filter
- Reject tiny regions: bbox.width < 50 or bbox.height < 30. 

(2) Content-based text rejection (PDF text layer)
- is_text_by_content(page, bbox):
  Extracts clipped text in bbox and checks:
  - English stopword ratio (prose indicator)
  - academic/bibliographic markers (hard reject)
  - gate keyword “rescue” (if gate tokens appear, do NOT reject as text) 

(3) Geometry-based text rejection (image-level)
- Render bbox crop to an image and run is_text_by_geometry(bgr):
  Uses OpenCV features:
  - extreme aspect ratio
  - pixel density thresholding
  - contour / “gate box” heuristic
  - Hough line detection for horizontal “wire-like” lines 

(4) Crop cleanup against caption leakage
- trim_bbox_using_pdf_text(page, expanded_bbox):
  Looks for large-width text blocks containing “Figure” or academic markers and trims top/bottom edges. 

Important design choice:
- These rules are cheap and reproducible; they reduce the number of garbage candidates reaching the CNN.

4.5 Stage E — CNN classification (circuit vs non-circuit)
---------------------------------------------------------

Files:
- circuit_classifier.py 
- extract_circuit_images.py 

Model:
- ResNet-18 with a 2-class head (circuit, non_circuit).
- Inference returns (label, probability) via predict_image(). 

Where it is applied:
- After the deterministic filters, each candidate crop is written to a temporary PNG
  and passed to predict_image(model, tmp_path). 

Thresholding:
- Candidates are accepted only if:
  - prob > 0.70 (very low prob rejection bucket)
  - label == "circuit"
  - prob >= prob_threshold (default 0.80) 

Output of this stage:
- Saved image file name:
  <arxiv_id>_page<page>_clusters<cluster_idx>.png
- Saved to: images_4/
- A metadata “record” describing where it came from, including expanded_bbox, caption_text, etc. 

Early stopping:
- extract_and_classify_clusters supports max_to_save so build_dataset.py can stop at exactly 250 records.  

4.6 Stage F — Page number + figure number + caption resolution
--------------------------------------------------------------

Page number:
- build_dataset.py passes doc_page_numbers (footer extracted) into extract_and_classify_clusters.
- If a printed footer number exists, it is used; otherwise, fallback is page_index + 1. 

Figure number + caption:
- detect_figure_number_and_caption(page, expanded_bbox)
  searches for caption-like lines below the figure first, then above, within a vertical margin,
  and uses multi-pattern regex to parse “Fig. 3(a) …”, “Figure 3.1 …”, “(b) …”, etc. 

Integer-only figure_number requirement:
- The assignment requires figure_number as integer. 
- Because papers contain labels like “3a”, “3.1”, or “(b)”, build_dataset.py normalizes them with:
  normalize_figure_number(fig):
  - int stays int
  - numeric + suffix (e.g., 3a) -> base*100 + suffix_index (3a => 301)
  - “3.1” becomes digits concatenated then int (3.1 => 31)
  - purely alphabetic -> -1 sentinel 

This normalization is deterministic and keeps schema compliance even when author figure labels are non-numeric.

4.7 Stage G — Description extraction + text_positions mapping
-------------------------------------------------------------

Files:
- build_dataset.py 
- text_layout_utils.py 

Goal:
- Fill:
  - "descriptions": list of short text parts relevant to the figure
  - "text_positions": list of (begin,end) spans in full_text for each description

Strategy:
1) Caption-first:
   - Use rec["caption_text"] from caption detection.
   - Clean with clean_description_text().
   - Reject math-like captions with is_math_like().
   - Reject irrelevant captions using calculate_semantic_relevance().  

2) Fallback: nearby paragraphs on same page
   - extract_paragraphs_around_figure(page, expanded_bbox, vertical_margin_pt=120)
   - select_description_sentences(paragraphs, max_sentences=3)
   This tries to pick a small number of the most “prose + circuit keyword” sentences near the figure. 

Position mapping:
- find_sentence_position_best(full_text, sentence_index, page_idx, target_text):
  - normalizes text variants
  - tries substring match within the same page’s sentences
  - otherwise falls back to fuzzy matching (difflib.SequenceMatcher)
  - returns (begin,end) offsets into full_text 

Only descriptions that successfully map to positions are kept (so every stored description has a stored span).

4.8 Stage H — Gate extraction + problem classification
------------------------------------------------------

File: gate_problem_extractor.py 
Called from: build_dataset.py

Gate extraction is “image-aware” and uses a strict authority hierarchy:

Step 1: OCR gates from the *circuit image* (highest priority)
- extract_gates_from_image(image_path):
  - Converts to grayscale, binarizes
  - Runs Tesseract OCR with "--psm 6"
  - Matches tokens to a gate lexicon
  - Accepts single-letter gates only if token is uppercase ("H", "X", "Z"), reducing variable confusion 

Step 2: If OCR finds no gates, fall back to strict text extraction
- extract_gates_from_text(caption + context + title):
  - Multi-character gates matched directly ("CNOT", "RX", "QFT", etc)
  - Single-letter gates only accepted with pre/post context keywords
    (e.g., "Pauli X gate", "X rotation") 

Final rule:
- If OCR returns >= 1 gate → OCR result wins (authoritative).
- Else → use text gates.
Returned list is deduplicated and canonicalized via lexicon mapping. 

Quantum problem / algorithm label:
- classify_problem(text):
  Tiered keyword matching:
  - Named algorithms first (Grover, Shor, VQE, QAOA, Deutsch-Jozsa)
  - Then functional labels (QFT, arithmetic, phase estimation)
  - Then broad categories (magic state injection, oracle gate implementation, etc)
  - Else "Unknown" 

Important limitation:
- This is keyword matching, not deep semantic inference. It will produce "Unknown" when keywords are absent or far away.

4.9 Stage I — Writing the dataset JSON + submission CSV
-------------------------------------------------------

File: build_dataset.py 

JSON format:
- Top-level: dict keyed by image filename.
- Each entry:
  {
    "arxiv_number": string,
    "page_number": int,
    "figure_number": int,
    "quantum_gates": [string],
    "quantum_problem": string,
    "descriptions": [string],
    "text_positions": [(begin,end), ...]
  }

Note:
- build_dataset.py writes JSON after each processed PDF (incremental progress saving).
- If metadata building fails for a saved image, the image is deleted to keep images/ and JSON consistent. 

Paper counts CSV:
- write_paper_counts_csv(tracking, exam_id)
  Writes paper_list_counts_4.csv with:
  - processed papers: extracted image_count
  - unprocessed papers: blank image_count
  - processed but none found: image_count = 0 

----------------------------------------------------------------------
5) Training the CNN classifier (optional)
----------------------------------------------------------------------

If you already have resnet18_circuit_classifier.pth, you do NOT need this.

File: train_circuit_classifier.py 

Expected folder structure under PROJECT_ROOT/data/:
- data/train/circuit/
- data/train/non_circuit/
- data/val/circuit/
- data/val/non_circuit/
- data/test/circuit/
- data/test/non_circuit/

Training procedure:
- Uses transfer learning with torchvision ResNet18 pretrained on ImageNet.
- Phase 1: train only final FC layer (NUM_EPOCHS_HEAD = 8)
- Phase 2: fine-tune layer4 + FC (NUM_EPOCHS_FINETUNE = 15) 
- Optionally prints a confusion matrix using scikit-learn if installed.
- Saves weights to resnet18_circuit_classifier.pth. 

Command:
$ python train_circuit_classifier.py

----------------------------------------------------------------------
6) Determinism and reproducibility notes
----------------------------------------------------------------------

This pipeline aims to be reproducible by construction:
- Fixed paper order from paper_list_4.txt
- Deterministic PDF processing loop (page-by-page, cluster-by-cluster)
- Deterministic rules for filtering, normalization, and thresholding

CNN inference determinism:
- circuit_classifier.py has set_deterministic(seed) which can be enabled by passing
  deterministic=True to load_circuit_model(). 
- Note: GPU determinism can still vary slightly depending on CUDA/cuDNN kernels.

Practical advice:
- If you want maximum reproducibility, run on CPU or enable deterministic inference.
- Keep the same versions of PyTorch/torchvision and the same model weights.

----------------------------------------------------------------------
7) Common failure modes + what the code does about them
----------------------------------------------------------------------

1) False positives (non-circuit figures saved as circuits)
- Caused by figures that share geometry with circuits (tables, block diagrams, line plots).
- Mitigation:
  - content/geometry text filters
  - conservative probability thresholds (0.80 default, plus >0.70 hard floor) 

2) False negatives (missed circuits)
- Tiny circuits, multi-panel composite figures, stylized notation, fragmented vector clusters.
- Mitigation:
  - cluster merging
  - padding/expansion
  - but small circuits can still be rejected by size filters or low confidence.

3) OCR gate noise
- Small fonts, symbols, poor rendering quality cause OCR mistakes.
- Mitigation:
  - OCR uses strict uppercase acceptance for single-letter gates
  - if OCR returns nothing, fallback is strict context-based text extraction. 

4) Figure numbering ambiguity
- Real papers use “(a)”, “S2”, “3.1b”, etc.
- Mitigation:
  - multi-pattern caption parsing + deterministic normalization to integer or -1 sentinel.

5) PDF parsing/rendering failures
- Some PDFs are malformed or contain unusual graphics states.
- Mitigation:
  - Exceptions are caught per paper; the pipeline skips failing PDFs and continues. 

----------------------------------------------------------------------
8) Key parameters you can tune (with consequences)
----------------------------------------------------------------------

In extract_circuit_images.py 
- ZOOM (default 3.5): higher increases crop resolution (better for OCR & CNN), but slower.
- MERGE_HORIZ_GAP / MERGE_VERT_GAP: affects clustering/merging of drawing clusters.
- Padding params (PAD_*): affects caption leakage vs circuit completeness.
- PROB_THRESHOLD_DEFAULT (0.80): higher → fewer false positives, more false negatives.

In build_dataset.py 
- TARGET_NUM_CIRCUITS (250): stop condition.
- normalize_figure_number(): encoding rules for non-integer labels.
- Description relevance thresholds (semantic relevance, fuzzy match ratios).

Be honest about tuning:
- Aggressive filtering can “clean” outputs but can also destroy recall and force processing many more PDFs.

----------------------------------------------------------------------
9) Verification checklist (what a grader can check fast)
----------------------------------------------------------------------

After running:
- images_4/ exists and contains PNGs.
- dataset_4.json exists and has exactly 250 entries (unless run stops early due to errors).
- paper_list_counts_4.csv exists and matches the processed order.
- For random entries in dataset_4.json:
  - image file exists in images_4/
  - page_number is integer
  - figure_number is integer (may be -1 sentinel)
  - descriptions length matches text_positions length
  - each (begin,end) span is valid inside full_text generated from the same PDF.

----------------------------------------------------------------------
10) Mapping to assignment requirements
----------------------------------------------------------------------

Assignment requirements and where they are implemented:

- Process papers in given order and stop after 250 circuit images
  - build_dataset.py: load_paper_list(), main() loop, TARGET_NUM_CIRCUITS 

- Save every valid image as PNG in images_<exam_id>
  - extract_circuit_images.py: pix.save(out_path) to images_4/ 

- JSON fields per image:
  - build_dataset.py: dataset[fname] dictionary structure 
  - gate_problem_extractor.py: quantum_gates + quantum_problem 
  - text_layout_utils.py + build_dataset.py: descriptions + text_positions 

- Add second column counts file paper_list_counts_<exam_id>.csv
  - build_dataset.py: write_paper_counts_csv() 

- Reproducible and generalizable method
  - Deterministic extraction rules + trained CNN classifier; no manual selection or hand-crafted dataset.

----------------------------------------------------------------------
11) Files (short reference)
----------------------------------------------------------------------

build_dataset.py
- Orchestrates everything: arXiv download, per-PDF processing, metadata extraction, JSON writing. 

extract_circuit_images.py
- Vector drawing clustering, candidate bbox merging, deterministic filters, CNN classification, image saving. 

text_layout_utils.py
- Footer page number extraction, figure number/caption parsing, caption cleaning, relevance scoring,
  paragraph selection, and full_text + sentence_index creation. 

gate_problem_extractor.py
- OCR-first gate detection + strict text fallback; hierarchical problem classification. 

circuit_classifier.py
- ResNet18 inference wrapper: consistent preprocessing, label mapping, predict_image API. 

train_circuit_classifier.py
- Optional training script: transfer learning, two-stage fine-tuning, model export. 

----------------------------------------------------------------------
12) Contact / notes
----------------------------------------------------------------------

This README describes the actual working pipeline as implemented in the provided scripts.
The report (NLP_W25_4.pdf) contains the methodology, error analysis, quality assessment,
and feasibility discussion, aligned with the assignment rubric.
