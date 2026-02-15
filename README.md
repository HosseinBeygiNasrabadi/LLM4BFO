# LLM4BFO
LLM4BFO is an interactive toolkit for the construction and evaluation of Basic Formal Ontology (BFO)-aligned ontologies using Large Language Models (LLMs).

The repository provides two desktop GUI applications that enable:
- **Ontology Builder:** Automated ontology generation (`.ttl`) from conceptual source files (PDF, Excel, or TTL/OWL)
- **Ontology Evaluation (Comparison):** Compares an LLM-generated ontology against a human ontology and exports the results to an Excel workbook (`.xlsx`).

# Repository Structure
```text
LLM4BFO/
│
├── codes/
│   │
│   ├── Ontology Builder/
│   │   ├── ontology_gui_multi_model.py
│   │   │   # GUI-1 launcher (Ontology Builder)
│   │   └── ontology_core_multi_model.py
│   │       # GUI-1 pipeline logic: input parsing → LLM call → TTL generation → validation
│   │
│   └── Ontology Evaluation/
│       ├── ontology_eval_gui.py
│       │   # GUI-2 launcher (Ontology Evaluation + XLSX export)
│       └── ontology_eval_core.py
│           # GUI-2 evaluation logic: class extraction → matching → similarity → hierarchy evaluation
│
├── testing examples/
│   ├── example_pdf_egg_boiling_recipe.pdf
│   ├── example_excel_vickers_hardness_test.xlsx
│   └── example_ontology_NFDIcore_v3.0.3.ttl
│
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

# 🛠 Installation Guide
## Requirements
- Python 3.10+ (the code uses modern type syntax like `dict | None`)
- A desktop environment that can run a Qt GUI (PyQt6)
- Runtime internet access is required for calling the selected LLM provider (OpenAI / Anthropic / Gemini) and downloading the SentenceTransformer model used by evaluation (`BAAI/bge-large-en-v1.5`).
- An LLM API key must be provided inside the GUI before ontology generation.

## 1) Clone the repository
```bash
git clone https://github.com/HosseinBeygiNasrabadi/LLM4BFO.git
cd LLM4BFO
```

## 2) Create and activate a virtual environment
Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

## Troubleshooting
- **Qt / platform plugin errors (Linux):** Some Linux setups require additional system Qt runtime packages. Install the missing Qt runtime packages for your distribution, then retry.

- **SentenceTransformer model download fails:** The evaluation uses SentenceTransformer model `BAAI/bge-large-en-v1.5`. If downloads are blocked, run in a network-enabled environment or pre-download the model.
  
- **`ModuleNotFoundError: No module named 'fitz'`**: PyMuPDF is missing:

```bash
pip install PyMuPDF
```

# ▶️ Running the Applications

## Run Ontology Builder

```bash
python ontology_gui_multi_model.py
```

<img width="570" height="415" alt="Screenshot 2026-02-15 at 13 37 20" src="https://github.com/user-attachments/assets/222e6d53-80b9-41b2-8201-fb990292412a" />


### Input Files
The GUI accepts different types of input files like (`.pdf`), (`.xlsx` / `.xls`), and Ontology file (`.ttl` / `.owl`). The pipeline parses them and extracts terms and definitions.

### User Inputs
The user should provide minimal information to run the GUI. This information includes LLM provider (`OpenAI` / `Anthropic` / `Gemini`), API key (label changes with provider), model name (defaults change with provider), and main ontology metadata like ontology title, label, description, version, domain, and ontology IRI.

### Output
- Output is a Turtle file (`.ttl`) written to the chosen path.
- The generated ontology includes:
  - A required ontology header with `owl:imports` for **BFO 2020**
  - Class IRIs derived from the provided **Ontology IRI** (base IRI) and a normalized label policy
  - Class blocks with `rdfs:label` and `skos:definition` fields (including “Original” and “Aristotelian” definitions)
    
## Run Ontology Evaluation (Comparison)

```bash
python ontology_eval_gui.py
```

<img width="570" height="241" alt="Screenshot 2026-02-15 at 13 38 42" src="https://github.com/user-attachments/assets/304eb07e-8ef5-48d0-9141-87c85a81cbdc" />


### Input Files
Human ontology (`.ttl` or `.owl`) and LLM-created ontology (`.ttl` or `.owl`)

### Outputs
The comparison results are shown in GUI, and the results are also exported as an Excel workbook next to the model ontology. The comparison topics include: header_comparison, groundtruth_classes, model_extracted_classes, label_level_matches, label_level_f1, definition_similarity, hierarchy_evaluation, hierarchy_f1, and evaluation_summary

# License
This project is released under the MIT License.

# Citation
If you use this software in academic work, please cite it using the CITATION.cff file included in this repository.

