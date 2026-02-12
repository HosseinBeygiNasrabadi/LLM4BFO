# Installation

## Requirements

- Python **3.10+** (the code uses modern type syntax like `dict | None`)
- A desktop environment that can run a Qt GUI (PyQt6)

Runtime internet access is required for:
- calling the selected LLM provider (OpenAI / Anthropic / Gemini)
- downloading the SentenceTransformer model used by evaluation (`BAAI/bge-large-en-v1.5`)

## Setup

### 1) Create and activate a virtual environment

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

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

## Run

### GUI-1: Ontology Builder

```bash
python ontology_gui_multi_model.py
```

### GUI-2: Ontology Evaluation

```bash
python ontology_eval_gui.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'fitz'`

PyMuPDF is missing:

```bash
pip install PyMuPDF
```

### Qt / platform plugin errors (Linux)

Some Linux setups require additional system Qt runtime packages. Install the missing Qt runtime packages for your distribution, then retry.

### SentenceTransformer model download fails

The evaluation uses SentenceTransformer model `BAAI/bge-large-en-v1.5`. If downloads are blocked, run in a network-enabled environment or pre-download the model.
