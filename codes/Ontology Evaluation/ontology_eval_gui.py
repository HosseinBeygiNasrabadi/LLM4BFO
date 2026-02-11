# ontology_eval_gui.py
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from PyQt6.QtWidgets import QSizePolicy

from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, Namespace
import pandas as pd
from ontology_eval_core import run_full_eval, parse_rdf_graph, select_local_class_nodes

DC = Namespace("http://purl.org/dc/elements/1.1/")
DCT = Namespace("http://purl.org/dc/terms/")


def extract_ontology_header(path: str):
    g = parse_rdf_graph(path)

    ont = None
    for s in g.subjects(RDF.type, OWL.Ontology):
        ont = s
        break

    def first(p):
        if ont is None:
            return ""
        for o in g.objects(ont, p):
            return str(o)
        return ""

    def all_vals(p):
        if ont is None:
            return ""
        return ", ".join(str(o) for o in g.objects(ont, p))

    return [
        ("rdfs:label", first(RDFS.label)),
        ("dc:contributor", first(DC.contributor)),
        ("dc:license", first(DC.license) or first(DCT.license)),
        ("dcterms:title", first(DCT.title)),
        ("dcterms:description", first(DCT.description)),
        ("owl:versionInfo", first(OWL.versionInfo)),
        ("owl:imports", all_vals(OWL.imports)),
    ]

class Worker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, h, m):
        super().__init__()
        self.h = h
        self.m = m

    def run(self):
        res = run_full_eval(self.h, self.m)
        self.finished.emit(res)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ontology Comparison")
        self.resize(1200, 900)

        central = QWidget()
        self.layout = QVBoxLayout(central)

        self.human = QLineEdit()
        self.model = QLineEdit()

        self.layout.addLayout(self._row("Human ontology (.ttl/.owl):", self.human))
        self.layout.addLayout(self._row("Model ontology (.ttl/.owl):", self.model))

        btn = QPushButton("Run comparison")
        btn.clicked.connect(self.run)
        self.layout.addWidget(btn)

        self.status = QLabel("")
        self.layout.addWidget(self.status)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self.scroll.setWidget(self.inner)

        self.layout.addWidget(self.scroll)
        self.setCentralWidget(central)

    def _row(self, label, edit):
        h = QHBoxLayout()
        h.addWidget(QLabel(label))
        h.addWidget(edit)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self.browse(edit))
        h.addWidget(b)
        return h

    def browse(self, edit):
        p, _ = QFileDialog.getOpenFileName(self, "", "", "TTL/OWL (*.ttl *.owl)")
        if p:
            edit.setText(p)

    def clear(self):
        while self.inner_layout.count():
            w = self.inner_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

    def add_table(self, title, df):
        self.inner_layout.addWidget(QLabel(f"<b>{title}</b>"))

        t = QTableWidget(df.shape[0], df.shape[1])
        t.setHorizontalHeaderLabels(df.columns)

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[r, c]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                t.setItem(r, c, item)

        # --- ENABLE USER RESIZING ---
        h = t.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        h.setStretchLastSection(True)
        h.setSectionsMovable(True)

        v = t.verticalHeader()
        v.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # --- ALLOW TABLE TO GROW / SHRINK ---
        t.setMinimumHeight(200)
        t.setMaximumHeight(16777215)
        t.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self.inner_layout.addWidget(t)

    def add_ontology_header_table(self, human_path: str, model_path: str):
        from rdflib import Graph, RDF, RDFS
        from rdflib.namespace import OWL

        def load_graph(p):
            return parse_rdf_graph(p)


        def find_ontology_node(g):
            for s in g.subjects(RDF.type, OWL.Ontology):
                return s
            return None

        def first_lit(g, s, p):
            if s is None:
                return ""
            for o in g.objects(s, p):
                return str(o)
            return ""

        def class_count(g):
            classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
            return len(classes)

        g_h = load_graph(human_path)
        g_m = load_graph(model_path)

        oh = find_ontology_node(g_h)
        om = find_ontology_node(g_m)

        rows = [
            ("rdfs:label", first_lit(g_h, oh, RDFS.label), first_lit(g_m, om, RDFS.label)),
            ("owl:versionInfo", first_lit(g_h, oh, OWL.versionInfo), first_lit(g_m, om, OWL.versionInfo)),
            ("Class count", str(class_count(g_h)), str(class_count(g_m))),
        ]


    def run(self):
        self.clear()

        self.status.setText("Running comparison… please wait")
        QApplication.processEvents()

        self.worker = Worker(self.human.text(), self.model.text())
        self.worker.finished.connect(self.display)
        self.worker.start()


    def display(self, res):
        self.status.setText("")

        header_h = extract_ontology_header(self.human.text())
        header_m = extract_ontology_header(self.model.text())

        rows = []
        for (k, hv), (_, mv) in zip(header_h, header_m):
            rows.append([k, hv, mv])

        def count_local_classes(path):
            _, nodes = select_local_class_nodes(path)
            return len(nodes)

        rows.append([
            "Class count",
            str(count_local_classes(self.human.text())),
            str(count_local_classes(self.model.text())),
        ])

        header_df = pd.DataFrame(rows, columns=["Property", "Human", "Model"])
        self.add_table("Ontology header comparison (Human vs Model)", header_df)
        self.add_table("Ground-truth classes", res["df_groundtruth"])
        self.add_table("Model-extracted classes", res["df_predicted"])
        self.add_table("Label-level matches", res["df_results"])
        self.add_table("Label-level F1", res["df_metrics"])
        self.add_table("Definition-level similarity", res["df_matched_def_sim"])
        self.add_table("Hierarchy-based evaluation", res["res_evaluation"])
        self.add_table("Hierarchy-based F1", res["res_f1"])

        # =========================================================
        # Evaluation summary table 
        # =========================================================

        # Name of ontology (prefer Model rdfs:label; fallback to file name)
        model_label = header_m[0][1] if header_m and header_m[0][0] == "rdfs:label" else ""
        if not model_label:
            model_label = os.path.basename(self.model.text())

        human_class_n = int(res["df_groundtruth"].shape[0])
        model_class_n = int(res["df_predicted"].shape[0])


        label_f1 = 0.0
        if isinstance(res.get("df_metrics"), pd.DataFrame) and not res["df_metrics"].empty:
            label_f1 = float(res["df_metrics"].iloc[0].get("F1", 0.0))

        hier_f1 = 0.0
        if isinstance(res.get("res_f1"), pd.DataFrame) and not res["res_f1"].empty:
            hier_f1 = float(res["res_f1"].iloc[0].get("F1", 0.0))

        def_sim_pct = 0.0
        df_defs = res.get("df_matched_def_sim")
        if isinstance(df_defs, pd.DataFrame) and not df_defs.empty and "definition_cosine_similarity" in df_defs.columns:
            sims = pd.to_numeric(df_defs["definition_cosine_similarity"], errors="coerce")
            m = sims.mean(skipna=True)
            if pd.notna(m):
                def_sim_pct = round(float(m) * 100.0, 2)

        summary_df = pd.DataFrame([{
            "Name of ontology": model_label,
            "Number of Human classes": human_class_n,
            "Number of LLM classes": model_class_n,
            "Classes Label Match F1": round(label_f1, 4),
            "Classes Definition Similarity (%)": def_sim_pct,
            "Classes Hierarchy Match F1": round(hier_f1, 4),
        }])

        self.add_table("Ontology classes evaluation summary", summary_df)

        # =========================================================
        # Export all results to XLSX 
        # =========================================================
        base = os.path.splitext(os.path.basename(self.model.text()))[0] or "ontology_eval"
        out_dir = os.path.join(os.path.dirname(self.model.text()) or ".", f"ontology_eval_xlsx_{base}")
        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(self.model.text()))[0] or "ontology_eval"
        out_dir = os.path.join(os.path.dirname(self.model.text()) or ".", f"ontology_eval_xlsx_{base}")

        out_path = os.path.join(out_dir, f"{base}_results.xlsx")

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            header_df.to_excel(writer, sheet_name="00_header_comparison", index=False)
            res["df_groundtruth"].to_excel(writer, sheet_name="01_groundtruth_classes", index=False)
            res["df_predicted"].to_excel(writer, sheet_name="02_model_extracted_classes", index=False)
            res["df_results"].to_excel(writer, sheet_name="03_label_level_matches", index=False)
            res["df_metrics"].to_excel(writer, sheet_name="04_label_level_f1", index=False)
            res["df_matched_def_sim"].to_excel(writer, sheet_name="05_definition_similarity", index=False)
            res["res_evaluation"].to_excel(writer, sheet_name="06_hierarchy_evaluation", index=False)
            res["res_f1"].to_excel(writer, sheet_name="07_hierarchy_f1", index=False)
            summary_df.to_excel(writer, sheet_name="08_evaluation_summary", index=False)

        self.status.setText(f"XLSX exported to: {out_path}")



def main():
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
