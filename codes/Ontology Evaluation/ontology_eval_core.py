# ontology_eval_core.py
import os
import re
import pandas as pd
from typing import Optional, List, Tuple, Set
from difflib import SequenceMatcher
from collections import defaultdict, Counter

from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS, Namespace
from owlready2 import get_ontology
from rdflib import Graph
from rdflib.namespace import DC, DCTERMS
from rdflib.collection import Collection
# =================================================
# CONSTANTS
# =================================================

BFO_PREFIX = "http://purl.obolibrary.org/obo/BFO_"
OBO = Namespace("http://purl.obolibrary.org/obo/")
VANN = Namespace("http://purl.org/vocab/vann/")
IAO_DEF = OBO["IAO_0000115"]  # definition predicate used by many OBO OWL files
SCHEMA = Namespace("http://schema.org/")

# =================================================
# BASIC HELPERS
# =================================================

def _looks_like_html_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        s = head.decode("utf-8", errors="ignore").lower()
        return ("<html" in s) or ("<!doctype html" in s) or ("<style" in s)
    except Exception:
        return False

def _parse_rdf_graph_with_format(path: str):
    # identical to ontology_core_multi_model._parse_rdf_graph
    if _looks_like_html_file(path):
        raise ValueError(
            f"File '{os.path.basename(path)}' looks like HTML, not Turtle/RDF."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext == ".ttl":
        fmts = ["turtle", "n3"]
    elif ext == ".owl":
        fmts = ["xml", "application/rdf+xml", "turtle", "n3"]
    else:
        fmts = ["xml", "turtle", "n3", "nt", "trig"]

    last_err = None
    for fmt in fmts:
        g = Graph()
        try:
            g.parse(path, format=fmt)
            return g, fmt
        except Exception as e:
            last_err = e

    g = Graph()
    try:
        g.parse(path)
        return g, "auto"
    except Exception as e:
        raise ValueError(
            f"Failed to parse RDF file '{path}'. Last error: {last_err}. Auto error: {e}"
        ) from e

def parse_rdf_graph(path: str) -> Graph:
    # keep the same public name used by GUI, but parse with the pipeline logic
    g, _ = _parse_rdf_graph_with_format(path)
    return g

def parse_rdf_graph_closure(path: str, import_map=None):
    g_all = Graph()
    seen = set()

    def _load(p: str):
        nonlocal g_all  

        if p in seen:
            return
        seen.add(p)

        g, _ = _parse_rdf_graph_with_format(p)
        g_all += g  

        for ont in g.subjects(RDF.type, OWL.Ontology):
            for imp in g.objects(ont, OWL.imports):
                imp_iri = str(imp)
                next_path = import_map.get(imp_iri, imp_iri) if import_map else imp_iri
                _load(next_path)

    _load(path)
    return g_all

OBO_PURL_DIR = "http://purl.obolibrary.org/obo/"
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")

def _local_name_from_uri(uri: str) -> str:
    # fragment after # or last path segment
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]

def _infer_obo_idspace_from_ontology_iri(ont_iri: str) -> Optional[str]:
    """
    If ontology IRI is like .../obo/label.owl
    """
    if not ont_iri:
        return None
    s = ont_iri.split("#", 1)[0]
    m = re.match(r"^http://purl\.obolibrary\.org/obo/([^/]+)\.(owl|ttl|rdf|xml|obo)$", s, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper().strip()

def _infer_obo_idspace_from_graph(g: Graph) -> Optional[str]:
    """
    Prefer explicit oboInOwl:idspace, else fall back to ontology IRI filename.
    """
    # 1) explicit idspace
    for ont in g.subjects(RDF.type, OWL.Ontology):
        for v in g.objects(ont, OBOINOWL.idspace):
            s = str(v).strip()
            if s:
                return s.upper()

    # 2) from ontology IRI
    for ont in g.subjects(RDF.type, OWL.Ontology):
        cand = _infer_obo_idspace_from_ontology_iri(str(ont))
        if cand:
            return cand

    return None

def _guess_namespace(uri: str) -> str:
    i_hash = uri.rfind("#")
    i_slash = uri.rfind("/")
    i = max(i_hash, i_slash)
    if i == -1:
        return uri
    return uri[: i + 1]


def _ontology_dir_namespace(ont_iri: str) -> Optional[str]:
    if not ont_iri:
        return None
    ont_iri = ont_iri.split("#", 1)[0]

    if re.search(r"\.(owl|ttl|rdf|xml|n3|nt|trig|obo)$", ont_iri, flags=re.IGNORECASE):
        if "/" in ont_iri:
            dir_ns = ont_iri.rsplit("/", 1)[0] + "/"

            # never use the shared OBO directory as "local"
            if dir_ns == OBO_PURL_DIR:
                return None

            return dir_ns
    return None


def _read_xml_base_if_present(path: str):
    try:
        with open(path, "rb") as f:
            head = f.read(20000)
        txt = head.decode("utf-8", errors="ignore")
        m = re.search(r'xml:base\s*=\s*"([^"]+)"', txt)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None

def _augment_namespaces_for_obo(local_ns_uris: Set[str]) -> Set[str]:
    # Do not automatically add the global /obo/ namespace
    return set(local_ns_uris)

def _infer_dominant_id_prefix(class_uris: List[str]):
    rx = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)_\d+$")
    prefixes = []
    for uri in class_uris:
        local = uri.rsplit("/", 1)[-1]
        m = rx.match(local)
        if m:
            prefixes.append(m.group(1))
    if not prefixes:
        return None
    c = Counter(prefixes)
    top, top_count = c.most_common(1)[0]
    return top if top_count >= 10 else None

def _get_local_namespace_uris_pipeline(graph: Graph) -> Set[str]:
    # identical to builder logic (including directory namespace)
    local_ns_uris: Set[str] = set()

    for ont in graph.subjects(RDF.type, OWL.Ontology):
        for ns in graph.objects(ont, VANN.preferredNamespaceUri):
            local_ns_uris.add(str(ns))

    if not local_ns_uris:
        prefixes = set()
        for ont in graph.subjects(RDF.type, OWL.Ontology):
            for lit in graph.objects(ont, VANN.preferredNamespacePrefix):
                prefixes.add(str(lit))

        if prefixes:
            for prefix, ns in graph.namespace_manager.namespaces():
                if prefix in prefixes:
                    local_ns_uris.add(str(ns))

    if not local_ns_uris:
        for ont in graph.subjects(RDF.type, OWL.Ontology):
            ont_iri = str(ont)
            if not ont_iri:
                continue

            stripped = ont_iri.rstrip("/#")
            if stripped:
                local_ns_uris.update({stripped, stripped + "/", stripped + "#"})

            # NEW (critical): directory namespace for file-like ontology IRIs
            dir_ns = _ontology_dir_namespace(ont_iri)
            if dir_ns:
                local_ns_uris.add(dir_ns)
                local_ns_uris.add(dir_ns + "#")

    if not local_ns_uris:
        ns_counts = Counter()
        for cls in set(graph.subjects(RDF.type, OWL.Class)):
            if cls.__class__.__name__ == "BNode":
                continue
            uri = str(cls)
            ns = _guess_namespace(uri)
            ns_counts[ns] += 1

        if ns_counts:
            max_count = max(ns_counts.values())
            for ns, count in ns_counts.items():
                if count >= max_count * 0.5 and count >= 1:
                    local_ns_uris.add(ns)

    if not local_ns_uris:
        for prefix, ns in graph.namespace_manager.namespaces():
            if prefix == "":
                local_ns_uris.add(str(ns))

    return local_ns_uris

def select_local_class_nodes(path: str) -> Tuple[Graph, List[URIRef]]:
    """
    Returns (graph, selected_nodes) using the SAME selection logic as
    ontology_core_multi_model.load_rdf_terms_with_hierarchy().
    """
    g, _ = _parse_rdf_graph_with_format(path)

    class_nodes: List[URIRef] = []
    class_uris: List[str] = []
    for cls in set(g.subjects(RDF.type, OWL.Class)):
        if cls.__class__.__name__ == "BNode":
            continue
        if not isinstance(cls, URIRef):
            continue
        class_nodes.append(cls)
        class_uris.append(str(cls))

    if not class_nodes:
        return g, []

    local_ns_uris = _get_local_namespace_uris_pipeline(g)

    if os.path.splitext(path)[1].lower() == ".owl":
        xml_base = _read_xml_base_if_present(path)
        if xml_base:
            stripped = xml_base.rstrip("/#")
            local_ns_uris.update({stripped, stripped + "/", stripped + "#"})

    # do NOT auto-add global /obo/
    local_ns_uris = _augment_namespaces_for_obo(local_ns_uris)

    dominant_prefix = _infer_dominant_id_prefix(class_uris)
    
    def _select_local_classes() -> List[URIRef]:
        # OBO IDSPACE-first
        idspace = _infer_obo_idspace_from_graph(g)
        if idspace:
            idspace_selected: List[URIRef] = []
            for cls in class_nodes:
                u = str(cls)
                if _local_name_from_uri(u).startswith(idspace + "_"):
                    idspace_selected.append(cls)
            if idspace_selected:
                return idspace_selected

        selected: List[URIRef] = []

        # 1) namespace-based selection
        if local_ns_uris:
            for cls in class_nodes:
                u = str(cls)
                if any(u.startswith(ns) for ns in local_ns_uris):
                    selected.append(cls)

        # If namespace selection succeeded, KEEP it
        if selected:
            return selected

        # fallback: all classes
        selected = class_nodes[:]

        # dominant prefix fallback ONLY when namespace selection failed
        if dominant_prefix:
            pref_selected_all: List[URIRef] = []
            for cls in class_nodes:
                local = _local_name_from_uri(str(cls))
                if local.startswith(dominant_prefix + "_"):
                    pref_selected_all.append(cls)

            if len(pref_selected_all) >= 10:
                return pref_selected_all

        return selected


    selected_nodes = _select_local_classes()
    selected_nodes = sorted(selected_nodes, key=lambda x: str(x))  # deterministic
    return g, selected_nodes


def label_like_pipeline(g: Graph, node) -> str:
    # identical to ontology_core_multi_model._collect_class_info() label part
    labels = [str(o) for o in g.objects(node, RDFS.label)]
    if labels:
        return labels[0]
    try:
        return node.n3(g.namespace_manager)
    except Exception:
        return str(node)

def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def class_count(g: Graph) -> int:
    return len(set(g.subjects(RDF.type, OWL.Class)))

def best_label(g: Graph, node: URIRef) -> str:
    for p in (RDFS.label, SKOS.prefLabel):
        for o in g.objects(node, p):
            return str(o)
    s = str(node)
    return s.split("#")[-1].split("/")[-1].replace("_", " ")

_DEF_NAME_RX = re.compile(
    r"(definition|description|textdefinition|gloss|documentation|explanation|note)",
    re.IGNORECASE
)

def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]

from rdflib.term import Literal

def discover_definition_predicates_human(g: Graph) -> List[URIRef]:
    bases = [
        SKOS.definition,
        IAO_DEF,
        DCTERMS.description,
        DC.description,
        SCHEMA.description,
    ]

    out: List[URIRef] = []
    seen: Set[URIRef] = set()

    def add(p: URIRef):
        if p not in seen:
            seen.add(p)
            out.append(p)

    for b in bases:
        add(b)

    # ---- broaden candidates to include predicates used with literals ----
    candidates = set(g.subjects(RDF.type, OWL.AnnotationProperty))
    candidates |= set(g.subjects(RDF.type, RDF.Property))

    # add any predicate that appears with a Literal object (even if untyped)
    for s, p, o in g.triples((None, None, None)):
        if isinstance(p, URIRef) and isinstance(o, Literal):
            candidates.add(p)
    # --------------------------------------------------------------------------

    def looks_definition_like(p: URIRef) -> bool:
        name = _local_name(str(p))
        labels = [str(lit) for lit in g.objects(p, RDFS.label)]
        return _DEF_NAME_RX.search(name) or any(_DEF_NAME_RX.search(l) for l in labels)

    for p in sorted([p for p in candidates if isinstance(p, URIRef)], key=lambda u: str(u)):
        if p not in seen and looks_definition_like(p):
            add(p)

    add(RDFS.comment)
    return out


def _guess_namespace(uri: str) -> str:
    i_hash = uri.rfind("#")
    i_slash = uri.rfind("/")
    i = max(i_hash, i_slash)
    if i == -1:
        return uri
    return uri[: i + 1]


def _read_xml_base_if_present(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            head = f.read(20000)
        txt = head.decode("utf-8", errors="ignore")
        m = re.search(r'xml:base\s*=\s*"([^"]+)"', txt)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None

def _infer_dominant_id_prefix(class_uris: List[str]) -> Optional[str]:
    rx = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)_\d+$")
    prefixes = []
    for uri in class_uris:
        local = uri.rsplit("/", 1)[-1]
        m = rx.match(local)
        if m:
            prefixes.append(m.group(1))
    if not prefixes:
        return None
    c = Counter(prefixes)
    top, top_count = c.most_common(1)[0]
    return top if top_count >= 10 else None


def _get_local_namespace_uris(graph: Graph) -> Set[str]:
    """
    SAME logic as builder:
      - preferredNamespaceUri (VANN)
      - preferredNamespacePrefix -> namespace_manager lookup
      - ontology IRI derived fallback
      - namespace frequency heuristic
      - empty prefix fallback
    """
    local_ns_uris: Set[str] = set()

    for ont in graph.subjects(RDF.type, OWL.Ontology):
        for ns in graph.objects(ont, VANN.preferredNamespaceUri):
            local_ns_uris.add(str(ns))

    if not local_ns_uris:
        prefixes = set()
        for ont in graph.subjects(RDF.type, OWL.Ontology):
            for lit in graph.objects(ont, VANN.preferredNamespacePrefix):
                prefixes.add(str(lit))

        if prefixes:
            for prefix, ns in graph.namespace_manager.namespaces():
                if prefix in prefixes:
                    local_ns_uris.add(str(ns))

    if not local_ns_uris:
        for ont in graph.subjects(RDF.type, OWL.Ontology):
            ont_iri = str(ont)
            if not ont_iri:
                continue

            stripped = ont_iri.rstrip("/#")
            if stripped:
                local_ns_uris.update({stripped, stripped + "/", stripped + "#"})

            # also consider the ontology directory namespace
            dir_ns = _ontology_dir_namespace(ont_iri)
            if dir_ns:
                local_ns_uris.add(dir_ns)
                local_ns_uris.add(dir_ns + "#")

    if not local_ns_uris:
        ns_counts = Counter()
        for cls in set(graph.subjects(RDF.type, OWL.Class)):
            if cls.__class__.__name__ == "BNode":
                continue
            uri = str(cls)
            ns = _guess_namespace(uri)
            ns_counts[ns] += 1

        if ns_counts:
            max_count = max(ns_counts.values())
            for ns, count in ns_counts.items():
                if count >= max_count * 0.5 and count >= 1:
                    local_ns_uris.add(ns)

    if not local_ns_uris:
        for prefix, ns in graph.namespace_manager.namespaces():
            if prefix == "":
                local_ns_uris.add(str(ns))

    return local_ns_uris

def select_local_class_nodes_like_builder(g: Graph, source_path: str) -> List[URIRef]:
    """
    Selection logic:
      1) namespace-based selection (derived local namespaces + xml:base for OWL)
      2) if namespace yields nothing -> start from all classes
      3) dominant prefix evaluated over ALL classes; if >=10 hits -> treat as local
    """
    class_nodes: List[URIRef] = []
    class_uris: List[str] = []

    for cls in set(g.subjects(RDF.type, OWL.Class)):
        if cls.__class__.__name__ == "BNode":
            continue
        if not isinstance(cls, URIRef):
            continue
        class_nodes.append(cls)
        class_uris.append(str(cls))

    if not class_nodes:
        return []

    local_ns_uris = _get_local_namespace_uris(g)

    ext = os.path.splitext(source_path)[1].lower()
    if ext == ".owl":
        xml_base = _read_xml_base_if_present(source_path)
        if xml_base:
            stripped = xml_base.rstrip("/#")
            local_ns_uris.update({stripped, stripped + "/", stripped + "#"})

    local_ns_uris = _augment_namespaces_for_obo(local_ns_uris)
    dominant_prefix = _infer_dominant_id_prefix(class_uris)

    selected: List[URIRef] = []

    # 1) namespace-based selection
    if local_ns_uris:
        for cls in class_nodes:
            u = str(cls)
            if any(u.startswith(ns) for ns in local_ns_uris):
                selected.append(cls)

    # If namespace selection succeeded, KEEP it (do not override with dominant prefix).
    if selected:
        return sorted(selected, key=lambda x: str(x))

    # If namespace selection finds nothing, start from all classes
    selected = class_nodes[:]  # fallback

    # Dominant prefix is a fallback ONLY (used when namespace selection failed)
    if dominant_prefix:
        pref_selected_all: List[URIRef] = []
        for cls in class_nodes:
            local = str(cls).rsplit("/", 1)[-1]
            if local.startswith(dominant_prefix + "_"):
                pref_selected_all.append(cls)

        if len(pref_selected_all) >= 10:
            return sorted(pref_selected_all, key=lambda x: str(x))

    return sorted(selected, key=lambda x: str(x))


def count_local_classes_like_builder(path: str) -> int:
    g = parse_rdf_graph(path)
    nodes = select_local_class_nodes_like_builder(g, path)
    return len(nodes)

# =================================================
# AUTHORITATIVE BFO RESOLUTION 
# =================================================

def first_bfo_from_graph(graph: Graph, class_label: str) -> Optional[str]:
    """
    Return the FIRST BFO ancestor encountered.
    Traverses:
      - rdfs:subClassOf
      - owl:equivalentClass
      - owl:intersectionOf (OWL class expressions)

    Skips ALL non-BFO classes (local, IAO, etc.).
    Stops at the first BFO class.
    """

    # 1. Find the class node by label
    start_nodes = []
    for cls in graph.subjects(RDF.type, OWL.Class):
        if norm(best_label(graph, cls)) == norm(class_label):
            start_nodes.append(cls)

    if not start_nodes:
        return None

    visited = set()
    stack = list(start_nodes)

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        # --- Case 1: node itself is BFO
        if isinstance(node, URIRef):
            uri = str(node)
            if uri.startswith(BFO_PREFIX):
                return f"bfo:{uri.split('_')[-1]}"

        # --- Case 2: rdfs:subClassOf
        for parent in graph.objects(node, RDFS.subClassOf):
            if isinstance(parent, URIRef):
                stack.append(parent)
            elif parent in graph.subjects(RDF.type, OWL.Class):
                stack.append(parent)

        # --- Case 3: owl:equivalentClass
        for eq in graph.objects(node, OWL.equivalentClass):
            stack.append(eq)

            # --- intersectionOf inside equivalentClass
            for inter in graph.objects(eq, OWL.intersectionOf):
                try:
                    for item in Collection(graph, inter):
                        stack.append(item)
                except Exception:
                    pass

        # --- Case 4: intersectionOf directly on node
        for inter in graph.objects(node, OWL.intersectionOf):
            try:
                for item in Collection(graph, inter):
                    stack.append(item)
            except Exception:
                pass

    return None

from rdflib.collection import Collection

def first_bfo_from_node(graph: Graph, start_node) -> Optional[str]:
    """
    Same traversal intent as first_bfo_from_graph(), but starts from a node (URI/BNode),
    so it does not depend on label matching.
    """
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if isinstance(node, URIRef):
            uri = str(node)
            if uri.startswith(BFO_PREFIX):
                return f"bfo:{uri.split('_')[-1]}"

        for parent in graph.objects(node, RDFS.subClassOf):
            stack.append(parent)

        for eq in graph.objects(node, OWL.equivalentClass):
            stack.append(eq)
            for inter in graph.objects(eq, OWL.intersectionOf):
                try:
                    for item in Collection(graph, inter):
                        stack.append(item)
                except Exception:
                    pass

        for inter in graph.objects(node, OWL.intersectionOf):
            try:
                for item in Collection(graph, inter):
                    stack.append(item)
            except Exception:
                pass

    return None

# =================================================
# GROUND-TRUTH CLASS EXTRACTION
# =================================================
def _guess_namespace(uri: str) -> str:
    i_hash = uri.rfind("#")
    i_slash = uri.rfind("/")
    return uri if max(i_hash, i_slash) == -1 else uri[: max(i_hash, i_slash) + 1]

def _get_local_namespace_uris(graph: Graph) -> Set[str]:
    local_ns = set()

    for ont in graph.subjects(RDF.type, OWL.Ontology):
        for ns in graph.objects(ont, VANN.preferredNamespaceUri):
            local_ns.add(str(ns))

    if not local_ns:
        ns_counts = Counter()
        for cls in graph.subjects(RDF.type, OWL.Class):
            if isinstance(cls, URIRef):
                ns_counts[_guess_namespace(str(cls))] += 1
        if ns_counts:
            max_count = max(ns_counts.values())
            for ns, c in ns_counts.items():
                if c >= max_count * 0.5:
                    local_ns.add(ns)

    return local_ns

def extract_local_classes_from_path(path: str) -> pd.DataFrame:
    g_local, nodes = select_local_class_nodes(path)

    # Useing closure graph for BFO traversal
    g_closure = parse_rdf_graph_closure(path, import_map=None)  

    rows = []
    for cls in nodes:
        if not isinstance(cls, URIRef):
            continue  

        lbl = label_like_pipeline(g_local, cls)  # keeping label source stable
        bfo = first_bfo_from_node(g_closure, cls)  # traverse on closure

        rows.append({
            "URI": str(cls),
            "Class": lbl,
            "BFO Parent": bfo
        })

    return pd.DataFrame(rows).drop_duplicates(subset=["URI"]) \
         .sort_values("Class").reset_index(drop=True)

def extract_ground_truth_classes(path: str) -> pd.DataFrame:
    return extract_local_classes_from_path(path)

def extract_model_classes(path: str) -> pd.DataFrame:
    return extract_local_classes_from_path(path)

# =================================================
# LABEL MATCHING 
# =================================================

def find_best_match(term, candidates, threshold=0.70):
    best, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, term, c).ratio()
        if score > best_score:
            best, best_score = c, score
    return (best, best_score) if best_score >= threshold else (None, best_score)

def _best_fuzzy_match(term: str, candidates: list[str]) -> tuple[str | None, float]:
    best = None
    best_score = 0.0
    for c in candidates:
        score = SequenceMatcher(None, term, c).ratio()
        if score > best_score:
            best, best_score = c, score
    return best, best_score

def label_match_eval(df_gt, df_pred, threshold=0.70):
    # keeping row identity (important if there are duplicates)
    gt_items = df_gt[["URI", "Class"]].reset_index(drop=True).copy()
    pr_items = df_pred[["URI", "Class"]].reset_index(drop=True).copy()

    gt_items["norm"] = gt_items["Class"].map(norm)
    pr_items["norm"] = pr_items["Class"].map(norm)

    # build all candidate pairs above threshold
    pairs = []
    for gi, gnorm in enumerate(gt_items["norm"]):
        for pj, pnorm in enumerate(pr_items["norm"]):
            score = SequenceMatcher(None, gnorm, pnorm).ratio()
            if score >= threshold:
                pairs.append((score, gi, pj))

    # sort by score descending (deterministic tie-breakers)
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    gt_used = set()
    pr_used = set()
    match_for_gt = [None] * len(gt_items)
    score_for_gt = [0.0] * len(gt_items)

    # assign best available pairs first
    for score, gi, pj in pairs:
        if gi in gt_used or pj in pr_used:
            continue
        gt_used.add(gi)
        pr_used.add(pj)
        match_for_gt[gi] = pj
        score_for_gt[gi] = score

    # build result rows in GT order
    rows = []
    for gi in range(len(gt_items)):
        gt_uri = str(gt_items.at[gi, "URI"])
        gt_norm = gt_items.at[gi, "norm"]

        pj = match_for_gt[gi]
        if pj is None:
            rows.append({
                "GT URI": gt_uri,
                "Ground Truth Term": gt_norm,
                "Pred URI": None,
                "Matched Term": None,
                "Label Match": "✗",
                "Score": round(score_for_gt[gi], 4),
            })
        else:
            pr_uri = str(pr_items.at[pj, "URI"])
            pr_norm = pr_items.at[pj, "norm"]
            rows.append({
                "GT URI": gt_uri,
                "Ground Truth Term": gt_norm,
                "Pred URI": pr_uri,
                "Matched Term": pr_norm,
                "Label Match": "✔",
                "Score": round(score_for_gt[gi], 4),
            })

    df = pd.DataFrame(rows)

    tp = (df["Label Match"] == "✔").sum()
    precision = tp / len(pr_items) if len(pr_items) else 0.0
    recall = tp / len(gt_items) if len(gt_items) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    df_metrics = pd.DataFrame([{
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    }])

    return df, df_metrics

# =================================================
# Definition Similarity
# =================================================

def definition_similarity(df_matches, human_ttl, model_ttl):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def load_defs_human(path):
        g = parse_rdf_graph(path)
        def_props = discover_definition_predicates_human(g)

        out = {}  # norm(label) -> {"original": str, "aristotelian": str, "any": str}

        for cls in g.subjects(RDF.type, OWL.Class):
            if not isinstance(cls, URIRef):
                continue

            lbl = best_label(g, cls)
            k = norm(lbl)

            defs = []
            for prop in def_props:
                for o in g.objects(cls, prop):
                    defs.append(str(o))

            original = ""
            arist = ""
            any_def = ""

            for d in defs:
                ds = d.strip()
                if not any_def and ds:
                    any_def = ds
                if ds.lower().startswith("original:") and not original:
                    original = ds
                if ds.lower().startswith("aristotelian:") and not arist:
                    arist = ds

            if not original:
                for d in defs:
                    ds = d.strip()
                    if ds and not ds.lower().startswith("aristotelian:"):
                        original = ds
                        break

            out[k] = {"original": original, "aristotelian": arist, "any": any_def}

        return out

    def load_defs_model(path):
        g = parse_rdf_graph(path)

        out = {}  # norm(label) -> {"original": str, "aristotelian": str, "any": str}

        for cls in g.subjects(RDF.type, OWL.Class):
            if not isinstance(cls, URIRef):
                continue

            lbl = best_label(g, cls)
            k = norm(lbl)

            defs = []
  
            for o in g.objects(cls, SKOS.definition):
                defs.append(str(o))

            original = ""
            arist = ""
            any_def = ""

            for d in defs:
                ds = d.strip()
                if not any_def and ds:
                    any_def = ds
                if ds.lower().startswith("original:") and not original:
                    original = ds
                if ds.lower().startswith("aristotelian:") and not arist:
                    arist = ds

            if not original:
                for d in defs:
                    ds = d.strip()
                    if ds and not ds.lower().startswith("aristotelian:"):
                        original = ds
                        break

            out[k] = {"original": original, "aristotelian": arist, "any": any_def}

        return out

    gt_defs = load_defs_human(human_ttl)
    pr_defs = load_defs_model(model_ttl)

    rows = []
    for _, r in df_matches.iterrows():
        if r["Label Match"] != "✔":
            continue

        gt = r["Ground Truth Term"]
        pr = r["Matched Term"]

        h = gt_defs.get(gt, {})
        m = pr_defs.get(pr, {})

        # Prefer Aristotelian if present, otherwise fall back to original/any
        d1 = h.get("aristotelian") or h.get("original") or h.get("any") or ""
        d2 = m.get("aristotelian") or m.get("original") or m.get("any") or ""


        if d1 and d2:
            v1 = model.encode([d1], normalize_embeddings=True)[0]
            v2 = model.encode([d2], normalize_embeddings=True)[0]
            sim = float(np.dot(v1, v2))
        else:
            sim = None

        rows.append({
            "Ground Truth Term": gt,
            "Matched Term": pr,
            "Human definition": d1,
            "Aristotelian generated": d2,
            "definition_cosine_similarity": None if sim is None else round(sim, 4)
        })

    return pd.DataFrame(rows)


# =================================================
# HIERARCHY EVALUATION 
# =================================================
def hierarchy_eval(df_matches, df_gt, df_pr):
    gt_bfo_by_uri = dict(zip(df_gt["URI"].astype(str), df_gt["BFO Parent"]))
    pr_bfo_by_uri = dict(zip(df_pr["URI"].astype(str), df_pr["BFO Parent"]))

    rows = []
    for _, r in df_matches.iterrows():
        gt_uri = r.get("GT URI")
        pr_uri = r.get("Pred URI")

        gt_bfo = gt_bfo_by_uri.get(str(gt_uri)) if gt_uri else None
        pr_bfo = pr_bfo_by_uri.get(str(pr_uri)) if pr_uri else None

        if r["Label Match"] != "✔":
            rows.append({
                "GT Class": r["Ground Truth Term"],
                "Pred Class": None,
                "GT BFO": gt_bfo,
                "Pred BFO": None,
                "Hierarchy Match": "NO_PREDICTION"
            })
            continue

        match = "✔" if (gt_bfo and pr_bfo and gt_bfo == pr_bfo) else "HIERARCHY_MISMATCH"

        rows.append({
            "GT Class": r["Ground Truth Term"],
            "Pred Class": r["Matched Term"],
            "GT BFO": gt_bfo,
            "Pred BFO": pr_bfo,
            "Hierarchy Match": match
        })

    df = pd.DataFrame(rows)

    tp = (df["Hierarchy Match"] == "✔").sum()
    fp = (df["Hierarchy Match"] == "HIERARCHY_MISMATCH").sum()
    fn = (df["Hierarchy Match"] == "NO_PREDICTION").sum()

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    df_f1 = pd.DataFrame([{
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4)
    }])

    return df, df_f1

# =================================================
# MAIN ENTRY 
# =================================================
def run_full_eval(human_ttl: str, model_ttl: str):
    df_gt = extract_ground_truth_classes(human_ttl)
    df_pr = extract_model_classes(model_ttl)

    df_matches, df_metrics = label_match_eval(df_gt, df_pr)
    df_hier, df_hier_f1 = hierarchy_eval(df_matches, df_gt, df_pr)
    df_defs = definition_similarity(df_matches, human_ttl, model_ttl)

    return {
        "df_groundtruth": df_gt,
        "df_predicted": df_pr,
        "df_results": df_matches,
        "df_metrics": df_metrics,
        "df_matched_def_sim": df_defs,
        "res_evaluation": df_hier,
        "res_f1": df_hier_f1,
    }
