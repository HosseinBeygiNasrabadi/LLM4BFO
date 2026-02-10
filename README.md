# LLM4BFO
LLM-based development of BFO-compliant ontologies

# Ontology Builder and Ontology Evaluation (GUI)

This repository contains two desktop GUI applications:

- **GUI-1: Ontology Builder (OD)**  
  Builds an ontology in Turtle (`.ttl`) from one input file:
  - PDF (standard text)
  - Excel (tabular text)
  - TTL/OWL (existing ontology)

- **GUI-2: Ontology Evaluation (Comparison)**  
  Compares an LLM-generated ontology against a human ontology and exports the results to an Excel workbook (`.xlsx`).


## Repository contents

- `ontology_gui_multi_model.py`  
  GUI-1 launcher (Ontology Builder)

- `ontology_core_multi_model.py`  
  GUI-1 pipeline logic (input parsing + LLM call + TTL generation + validation)

- `ontology_eval_gui.py`  
  GUI-2 launcher (Ontology Evaluation + XLSX export)

- `ontology_eval_core.py`  
  GUI-2 evaluation logic (class extraction + matching + similarity + hierarchy evaluation)


## GUI-1: Ontology Builder (OD)

### Inputs
The GUI accepts one input file:

1) **PDF (`.pdf`)**
- The pipeline extracts **Section 3 (Terms and definitions)**.
- It expects term identifiers starting with `3...` such as `3.1`, `3.2.1`, etc.
- Each term is extracted as:
  - `num` (the section number, e.g., `3.2.1`)
  - `label`
  - `definition` (inline after `:` or dash, and/or block text below)

2) **Excel (`.xlsx` / `.xls`)**
- The first row must be a header row.
- Required header column: **`Term`**
- Optional header column: **`Definition`**
- Each non-empty `Term` row becomes a class.
- Terms are assigned sequential numeric IDs (`1`, `2`, `3`, ...) internally.

3) **Ontology file (`.ttl` / `.owl`)**
- The pipeline parses RDF and extracts **local** `owl:Class` nodes (local namespace is inferred from ontology metadata and/or dominant namespaces).
- It also extracts local subclass relations to build an internal hierarchy.

### GUI fields (exactly as implemented)
- Provider: `OpenAI` / `Anthropic` / `Gemini`
- API key (label changes with provider)
- Model name (defaults change with provider)
- Metadata:
  - Title
  - Label
  - Description
  - Version
  - Domain
  - Ontology IRI

### Output
- Output is a Turtle file (`.ttl`) written to the chosen path.
- The generated ontology includes:
  - A required ontology header with `owl:imports` for **BFO 2020**
  - Class IRIs derived from the provided **Ontology IRI** (base IRI) and a normalized label policy
  - Class blocks with `rdfs:label` and `skos:definition` fields (including “Original” and “Aristotelian” definitions)

### How to run GUI-1
```bash
python ontology_gui_multi_model.py
```

## GUI-2: Ontology Evaluation (Comparison)

### Inputs

- Human ontology: .ttl or .owl

- Model ontology: .ttl or .owl

### What is computed

- Ontology header comparison (basic metadata inspection)

- Extraction of class sets from both ontologies

- Label-level fuzzy matching (SequenceMatcher ratio, threshold = 0.70)

- Definition similarity (sentence embeddings using BAAI/bge-large-en-v1.5)

- Hierarchy comparison based on extracted BFO parent information from matched classes

- Precision / Recall / F1 for:

  - label-level matching

  - hierarchy-level matching

### Output

The GUI exports an Excel workbook next to the model ontology, in:

- Directory: ontology_eval_xlsx_<model_basename>/

- File: <model_basename>_results.xlsx

The workbook contains these sheets (exact names):

- 00_header_comparison

- 01_groundtruth_classes

- 02_model_extracted_classes

- 03_label_level_matches

- 04_label_level_f1

- 05_definition_similarity

- 06_hierarchy_evaluation

- 07_hierarchy_f1

- 08_evaluation_summary

How to run GUI-2
```bash
python ontology_eval_gui.py
```

## Notes

- GUI-2 downloads an embedding model (BAAI/bge-large-en-v1.5) via sentence-transformers the first time it runs (internet required).

- API keys are entered in the GUI. The code does not implement persistent secret storage in this repository.
