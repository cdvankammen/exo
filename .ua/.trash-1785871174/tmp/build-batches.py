#!/usr/bin/env python3
"""
Deterministic batch graph builder for the understand-anything pipeline.

Converts extract-structure.mjs results + batchImportData into GraphNode/GraphEdge
JSON for batches that could not be dispatched to LLM file-analyzer subagents
(upstream provider outage). Mirrors the file-analyzer.md protocol:

- Node types by fileCategory: code->file, config->config, docs->document,
  infra->service/pipeline/resource, data->table/schema/endpoint, script->file,
  markup->file
- function:/class: nodes for significant functions (10+ lines or exported) and
  classes (2+ methods or 20+ lines or exported)
- Edges: contains (1.0), imports (0.7, 1:1 with batchImportData), exports (0.8),
  calls (0.8, intra-file callGraph), tested_by (0.5, test->production),
  configures/documents/deploys/triggers for non-code files
- Node IDs use exact prefixes; never project-name prefixed; never bare paths
"""
import json
import glob
import os
import sys

PROJECT_ROOT = "/Users/chris/Documents/GitHub/exo"
UA_DIR = os.path.join(PROJECT_ROOT, ".ua")
INTERMEDIATE = os.path.join(UA_DIR, "intermediate")
TMP = os.path.join(UA_DIR, "tmp")

# fileCategory -> node type
CATEGORY_NODE_TYPE = {
    "code": "file",
    "config": "config",
    "docs": "document",
    "script": "file",
    "markup": "file",
}

# infra sub-types by path/language
def infra_node_type(path, language):
    p = path.lower()
    if any(x in p for x in (".github/workflows", ".gitlab-ci", "jenkinsfile", ".circleci")):
        return "pipeline"
    if any(x in p for x in (".tf", ".tfvars", "cloudformation", "vagrantfile")):
        return "resource"
    return "service"  # Dockerfile, docker-compose, k8s

def data_node_type(path, language):
    p = path.lower()
    if language in ("graphql", "protobuf", "prisma") or p.endswith(".graphql") or p.endswith(".proto") or p.endswith(".prisma"):
        return "schema"
    if language in ("openapi", "swagger") or "openapi" in p or "swagger" in p:
        return "endpoint"
    return "table"  # SQL

def node_type_for(path, language, file_category):
    if file_category == "infra":
        return infra_node_type(path, language)
    if file_category == "data":
        return data_node_type(path, language)
    return CATEGORY_NODE_TYPE.get(file_category, "file")

def node_id_for(path, node_type):
    return f"{node_type}:{path}"

def complexity_for(non_empty_lines, func_count, class_count):
    if non_empty_lines < 50 and func_count <= 2 and class_count == 0:
        return "simple"
    if non_empty_lines <= 200:
        return "moderate"
    return "complex"

def build_batch(batch_index, batch_files, batch_import_data, extract_results):
    nodes = []
    edges = []
    node_ids = set()

    # Map path -> extraction result
    results_by_path = {}
    for r in extract_results:
        results_by_path[r["path"]] = r

    # First pass: create file-level nodes
    for f in batch_files:
        path = f["path"]
        language = f["language"]
        file_category = f["fileCategory"]
        size_lines = f.get("sizeLines", 0)
        ntype = node_type_for(path, language, file_category)
        nid = node_id_for(path, ntype)

        res = results_by_path.get(path, {})
        non_empty = res.get("nonEmptyLines", size_lines)
        funcs = res.get("functions", [])
        classes = res.get("classes", [])
        exports = res.get("exports", [])

        # Determine tags
        tags = []
        base = os.path.basename(path)
        if ntype == "config":
            tags = ["configuration"]
        elif ntype == "document":
            tags = ["documentation"]
        elif ntype == "pipeline":
            tags = ["ci-cd", "deployment"]
        elif ntype == "service":
            tags = ["containerization", "infrastructure"]
        elif ntype == "resource":
            tags = ["infrastructure", "deployment"]
        elif ntype == "schema":
            tags = ["schema-definition", "api-schema"]
        elif ntype == "table":
            tags = ["database", "migration"]
        elif ntype == "endpoint":
            tags = ["api-schema"]
        else:
            # code file
            if base.startswith("test_") or ".test." in base or ".spec." in base or base.startswith("test-"):
                tags.append("test")
            if base == "__init__.py":
                tags.append("entry-point")
                if len(exports) > 0 or len(funcs) == 0:
                    tags.append("barrel")
            if base in ("main.py", "__main__.py", "manage.py"):
                tags.append("entry-point")
            if len(exports) > 0 and len(funcs) == 0 and len(classes) == 0:
                tags.append("barrel")
            if any("Handler" in e["name"] or "Controller" in e["name"] for e in exports):
                tags.append("api-handler")
            if not tags:
                tags.append("module")
        # dedupe, cap at 5
        seen = set()
        tags = [t for t in tags if not (t in seen or seen.add(t))][:5]

        summary = f"{base} — {file_category} file in the exo project."
        if ntype == "config":
            summary = f"Configuration file {path} controlling project settings."
        elif ntype == "document":
            summary = f"Documentation file {path}."
        elif ntype == "pipeline":
            summary = f"CI/CD pipeline definition {path}."
        elif ntype == "service":
            summary = f"Container/service definition {path}."
        elif ntype == "schema":
            summary = f"Schema definition {path}."
        elif ntype == "table":
            summary = f"Database table definition {path}."
        elif ntype == "endpoint":
            summary = f"API endpoint specification {path}."
        elif funcs or classes:
            summary = f"Defines {len(funcs)} function(s) and {len(classes)} class(es) for {path}."

        node = {
            "id": nid,
            "type": ntype,
            "name": base,
            "filePath": path,
            "summary": summary,
            "tags": tags,
            "complexity": complexity_for(non_empty, len(funcs), len(classes)),
        }
        nodes.append(node)
        node_ids.add(nid)

    # Second pass: function/class nodes + contains/exports edges
    for f in batch_files:
        path = f["path"]
        ntype = node_type_for(path, f["language"], f["fileCategory"])
        nid = node_id_for(path, ntype)
        res = results_by_path.get(path, {})
        funcs = res.get("functions", [])
        classes = res.get("classes", [])
        exports = res.get("exports", [])
        exported_names = {e["name"] for e in exports}

        for fn in funcs:
            name = fn["name"]
            start = fn.get("startLine", 0)
            end = fn.get("endLine", 0)
            length = end - start + 1
            is_exported = name in exported_names
            if length < 10 and not is_exported:
                continue
            fid = f"function:{path}:{name}"
            nodes.append({
                "id": fid,
                "type": "function",
                "name": name,
                "filePath": path,
                "lineRange": [start, end],
                "summary": f"Function {name} in {path}.",
                "tags": ["function"],
                "complexity": "simple" if length < 50 else ("moderate" if length <= 200 else "complex"),
            })
            node_ids.add(fid)
            edges.append({"source": nid, "target": fid, "type": "contains", "direction": "forward", "weight": 1.0})
            if is_exported:
                edges.append({"source": nid, "target": fid, "type": "exports", "direction": "forward", "weight": 0.8})

        for cls in classes:
            name = cls["name"]
            methods = cls.get("methods", [])
            start = cls.get("startLine", 0)
            end = cls.get("endLine", 0)
            length = end - start + 1
            is_exported = name in exported_names
            if len(methods) < 2 and length < 20 and not is_exported:
                continue
            cid = f"class:{path}:{name}"
            nodes.append({
                "id": cid,
                "type": "class",
                "name": name,
                "filePath": path,
                "lineRange": [start, end],
                "summary": f"Class {name} in {path}.",
                "tags": ["class"],
                "complexity": "simple" if length < 50 else ("moderate" if length <= 200 else "complex"),
            })
            node_ids.add(cid)
            edges.append({"source": nid, "target": cid, "type": "contains", "direction": "forward", "weight": 1.0})
            if is_exported:
                edges.append({"source": nid, "target": cid, "type": "exports", "direction": "forward", "weight": 0.8})

    # Third pass: imports edges (1:1 with batchImportData)
    import_count = 0
    for f in batch_files:
        path = f["path"]
        ntype = node_type_for(path, f["language"], f["fileCategory"])
        nid = node_id_for(path, ntype)
        if ntype != "file":
            continue
        for target in batch_import_data.get(path, []):
            edges.append({"source": nid, "target": f"file:{target}", "type": "imports", "direction": "forward", "weight": 0.7})
            import_count += 1

    # Fourth pass: intra-file calls edges (callGraph)
    for f in batch_files:
        path = f["path"]
        ntype = node_type_for(path, f["language"], f["fileCategory"])
        nid = node_id_for(path, ntype)
        res = results_by_path.get(path, {})
        for call in res.get("callGraph", []):
            caller = call.get("caller")
            callee = call.get("callee")
            if not caller or not callee:
                continue
            # Only emit calls between functions defined in this file
            caller_id = f"function:{path}:{caller}"
            callee_id = f"function:{path}:{callee}"
            if caller_id in node_ids and callee_id in node_ids:
                edges.append({"source": caller_id, "target": callee_id, "type": "calls", "direction": "forward", "weight": 0.8})

    # Fifth pass: tested_by edges (test files -> production files they import)
    for f in batch_files:
        path = f["path"]
        base = os.path.basename(path)
        is_test = base.startswith("test_") or ".test." in base or ".spec." in base or "tests" in path.split("/")
        if not is_test:
            continue
        ntype = node_type_for(path, f["language"], f["fileCategory"])
        nid = node_id_for(path, ntype)
        for target in batch_import_data.get(path, []):
            # production file -> test file direction
            edges.append({"source": f"file:{target}", "target": nid, "type": "tested_by", "direction": "forward", "weight": 0.5})

    # Sixth pass: non-code edges (configures/documents/deploys/triggers)
    for f in batch_files:
        path = f["path"]
        ntype = node_type_for(path, f["language"], f["fileCategory"])
        nid = node_id_for(path, ntype)
        if ntype == "config":
            # configures the most relevant code files it imports/mentions
            for target in batch_import_data.get(path, []):
                edges.append({"source": nid, "target": f"file:{target}", "type": "configures", "direction": "forward", "weight": 0.6})
        elif ntype == "document":
            for target in batch_import_data.get(path, []):
                edges.append({"source": nid, "target": f"file:{target}", "type": "documents", "direction": "forward", "weight": 0.5})
        elif ntype == "pipeline":
            for target in batch_import_data.get(path, []):
                edges.append({"source": nid, "target": f"file:{target}", "type": "triggers", "direction": "forward", "weight": 0.6})
        elif ntype == "service":
            for target in batch_import_data.get(path, []):
                edges.append({"source": nid, "target": f"file:{target}", "type": "deploys", "direction": "forward", "weight": 0.7})

    return {"nodes": nodes, "edges": edges, "import_count": import_count}


def main():
    batches = json.load(open(os.path.join(INTERMEDIATE, "batches.json")))
    existing = set()
    for f in glob.glob(os.path.join(INTERMEDIATE, "batch-*.json")):
        base = os.path.basename(f)
        # handles batch-<i>.json and batch-<i>-part-<k>.json
        stem = base[len("batch-"):-len(".json")]
        idx = stem.split("-")[0]
        existing.add(int(idx))

    missing = [b["batchIndex"] for b in batches["batches"] if b["batchIndex"] not in existing]
    print(f"Existing batches: {sorted(existing)}")
    print(f"Missing batches: {missing}")

    for i in missing:
        b = next(x for x in batches["batches"] if x["batchIndex"] == i)
        batch_files = b["files"]
        batch_import_data = b.get("batchImportData", {})
        results_path = os.path.join(TMP, f"ua-file-extract-results-{i}.json")
        if not os.path.exists(results_path):
            print(f"WARN: no extraction results for batch {i}, skipping")
            continue
        extract_results = json.load(open(results_path)).get("results", [])
        graph = build_batch(i, batch_files, batch_import_data, extract_results)
        out_path = os.path.join(INTERMEDIATE, f"batch-{i}.json")
        with open(out_path, "w") as f:
            json.dump({"nodes": graph["nodes"], "edges": graph["edges"]}, f, indent=1)
        print(f"batch-{i}.json: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, imports={graph['import_count']}")

    print("DONE")


if __name__ == "__main__":
    main()