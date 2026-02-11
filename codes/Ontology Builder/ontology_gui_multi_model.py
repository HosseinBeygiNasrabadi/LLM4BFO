# ontology_gui_multi.py
# ------------------------------------------------------------
# GUI for multi-input ontology builder (PDF / TTL / OWL / Excel)
# with provider selection: OpenAI / Anthropic / Gemini
# ------------------------------------------------------------

import sys
from datetime import date

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QPlainTextEdit,
    QMessageBox,
    QStatusBar,
    QComboBox,
)
from PyQt6.QtGui import QPalette, QColor, QTextOption, QFontDatabase

from ontology_core_multi_model import run_pipeline


def apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#0e1116"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e6edf3"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#161b22"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e6edf3"))
    app.setPalette(pal)


class Worker(QThread):
    line = pyqtSignal(str)
    done = pyqtSignal(int)

    def __init__(self, input_path, ttl_out_path, provider, api_key, model_name, meta):
        super().__init__()
        self.input_path = input_path
        self.ttl_out_path = ttl_out_path
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.meta = meta

    def run(self):
        try:
            out = run_pipeline(
                input_path=self.input_path,
                ttl_out=self.ttl_out_path,
                provider=self.provider,
                api_key=self.api_key,
                model_name=self.model_name,
                meta=self.meta,
            )
            for line in out.splitlines():
                self.line.emit(line)
            self.done.emit(0)
        except Exception as e:
            self.line.emit(f"[Error] {e}")
            self.done.emit(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ontology Builder (Multi-input GUI)")
        self.resize(900, 750)

        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Input file (PDF/TTL/OWL/Excel)
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input (PDF / TTL / OWL / Excel):"))
        self.in_edit = QLineEdit()
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self.choose_input)
        in_row.addWidget(self.in_edit, 1)
        in_row.addWidget(btn_in)
        layout.addLayout(in_row)

        # Output TTL
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output TTL:"))
        self.out_edit = QLineEdit("generated_ontology.ttl")
        btn_out = QPushButton("Choose…")
        btn_out.clicked.connect(self.choose_output)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        # Provider selector
        prov_row = QHBoxLayout()
        prov_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["OpenAI", "Anthropic", "Gemini"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        prov_row.addWidget(self.provider_combo, 1)
        layout.addLayout(prov_row)

        # API key (label changes depending on provider)
        api_row = QHBoxLayout()
        self.api_label = QLabel("OpenAI API key:")
        api_row.addWidget(self.api_label)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        api_row.addWidget(self.key_edit, 1)
        layout.addLayout(api_row)

        # Model name (provider-specific)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model name:"))
        self.model_edit = QLineEdit("gpt-5.2")
        model_row.addWidget(self.model_edit, 1)
        layout.addLayout(model_row)

        # Metadata
        def meta_row(label, default=""):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            edit = QLineEdit(default)
            row.addWidget(edit, 1)
            layout.addLayout(row)
            return edit

        today_ver = date.today().strftime("%Y.%m.%d") + ".LLM"
        self.title_edit = meta_row("Title:", "Iso standard ontology")
        self.label_edit = meta_row("Label:", "onto")
        self.desc_edit = meta_row("Description:", "Ontology created from input classes and definitions.")
        self.version_edit = meta_row("Version:", today_ver)
        self.domain_edit = meta_row("Domain:", "Iso standard")
        self.iri_edit = meta_row("Ontology IRI:", "https://github.com/ISE-FIZKarlsruhe/LLM4BFO/onto")

        # buttons
        ctrl_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_job)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(lambda: self.console.clear())
        ctrl_row.addWidget(self.run_btn)
        ctrl_row.addWidget(self.clear_btn)
        layout.addLayout(ctrl_row)

        # console
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.console.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(self.console, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # init provider defaults
        self.on_provider_changed(self.provider_combo.currentText())

    def on_provider_changed(self, provider: str):
        provider = (provider or "").strip()
        if provider == "Anthropic":
            self.api_label.setText("Anthropic API key:")
            if self.model_edit.text().strip() == "" or self.model_edit.text().strip().startswith("gpt-"):
                self.model_edit.setText("claude-sonnet-4-5")
        elif provider == "Gemini":
            self.api_label.setText("Gemini API key:")
            if self.model_edit.text().strip() == "" or self.model_edit.text().strip().startswith("gpt-"):
                self.model_edit.setText("gemini-2.5-pro")
        else:
            self.api_label.setText("OpenAI API key:")
            if self.model_edit.text().strip() == "" or self.model_edit.text().strip().startswith("claude") or self.model_edit.text().strip().startswith("gemini"):
                self.model_edit.setText("gpt-5.2")

    def choose_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input file",
            "",
            "All supported (*.pdf *.ttl *.owl *.xlsx *.xls);;"
            "PDF files (*.pdf);;"
            "RDF files (*.ttl *.owl);;"
            "Excel files (*.xlsx *.xls);;"
            "All files (*.*)",
        )
        if path:
            self.in_edit.setText(path)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output TTL", "generated_ontology.ttl", "TTL files (*.ttl)"
        )
        if path:
            self.out_edit.setText(path)

    def run_job(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Running", "A job is already running.")
            return

        inp = self.in_edit.text().strip()
        outp = self.out_edit.text().strip()
        provider = self.provider_combo.currentText().strip()
        key = self.key_edit.text().strip()
        model = self.model_edit.text().strip()

        if not inp or not key:
            QMessageBox.warning(self, "Missing data", "Please choose an input file and enter API key.")
            return
        if not model:
            QMessageBox.warning(self, "Missing data", "Please enter a model name.")
            return

        meta = {
            "title": self.title_edit.text().strip(),
            "label": self.label_edit.text().strip(),
            "description": self.desc_edit.text().strip(),
            "version": self.version_edit.text().strip(),
            "domain": self.domain_edit.text().strip(),
            "ontology_iri": self.iri_edit.text().strip(),
        }

        self.console.appendPlainText("Starting pipeline...\n")

        self.worker = Worker(inp, outp, provider, key, model, meta)
        self.worker.line.connect(self.console.appendPlainText)
        self.worker.done.connect(self.on_worker_done)
        self.worker.start()

    def on_worker_done(self, code: int):
        self.status.showMessage(f"Finished with code {code}", 3000)
        self.worker = None


def main():
    app = QApplication(sys.argv)
    apply_dark_palette(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
