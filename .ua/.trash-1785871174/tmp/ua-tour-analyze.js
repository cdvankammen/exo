#!/usr/bin/env node
"use strict";
/**
 * ua-tour-analyze.js — Graph topology analysis for tour-builder (Phase 1).
 * Usage: node ua-tour-analyze.js <input.json> <output.json>
 * Exit 0 on success, 1 on fatal error.
 */
const fs = require("fs");

function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!inputPath || !outputPath) {
    console.error("usage: node ua-tour-analyze.js <input.json> <output.json>");
    process.exit(1);
  }
  const raw = fs.readFileSync(inputPath, "utf8");
  const data = JSON.parse(raw);
  const nodes = data.nodes || [];
  const edges = data.edges || [];
  const layers = data.layers || [];

  const byId = new Map();
  for (const n of nodes) byId.set(n.id, n);
  const nodeIds = new Set(nodes.map((n) => n.id));

  // ---- Fan-in / Fan-out over edges whose endpoints are both file-level ----
  const fanIn = new Map();
  const fanOut = new Map();
  for (const id of nodeIds) {
    fanIn.set(id, 0);
    fanOut.set(id, 0);
  }
  // adjacency for BFS / clusters: forward (imports/calls) and undirected
  const fwd = new Map(); // id -> Set(ids)  (imports + calls)
  const und = new Map(); // id -> Set(ids)  (all file-level edges)
  for (const id of nodeIds) {
    fwd.set(id, new Set());
    und.set(id, new Set());
  }
  for (const e of edges) {
    const { source, target, type } = e;
    if (!nodeIds.has(source) || !nodeIds.has(target)) continue;
    fanIn.set(target, fanIn.get(target) + 1);
    fanOut.set(source, fanOut.get(source) + 1);
    und.get(source).add(target);
    und.get(target).add(source);
    if (type === "imports" || type === "calls") fwd.get(source).add(target);
  }

  const rankList = (map) =>
    [...map.entries()]
      .map(([id, v]) => ({ id, [map === fanIn ? "fanIn" : "fanOut"]: v, name: byId.get(id).name }))
      .sort((a, b) => (map === fanIn ? b.fanIn - a.fanIn : b.fanOut - a.fanOut));

  const fanInRanking = rankList(fanIn).slice(0, 20);
  const fanOutRanking = rankList(fanOut).slice(0, 20);

  // ---- Entry point candidates ----
  const ENTRY_NAMES = new Set([
    "index.ts","index.js","main.ts","main.js","app.ts","app.js","server.ts",
    "server.js","mod.rs","main.go","main.py","main.rs","manage.py","app.py",
    "wsgi.py","asgi.py","run.py","__main__.py","Application.java","Main.java",
    "Program.cs","config.ru","index.php","App.swift","Application.kt",
    "main.cpp","main.c",
  ]);
  const scores = [];
  for (const n of nodes) {
    const id = n.id;
    let score = 0;
    if (n.type === "document") {
      if (n.name === "README.md") score += 5;
      else if (n.filePath && n.filePath.endsWith(".md") && n.filePath.split("/").length <= 2) score += 2;
    } else {
      const base = n.name || "";
      if (ENTRY_NAMES.has(base)) score += 3;
      // root or one level deep (e.g. src/main.py)
      if (n.filePath) {
        const parts = n.filePath.split("/");
        if (parts.length <= 2) score += 1;
      }
      const total = nodes.length;
      const fo = fanOut.get(id) || 0;
      const fi = fanIn.get(id) || 0;
      const sortedFO = [...fanOut.values()].sort((a, b) => b - a);
      const sortedFI = [...fanIn.values()].sort((a, b) => a - b);
      const top10FO = sortedFO[Math.max(0, Math.floor(total * 0.1) - 1)];
      const bottom25FI = sortedFI[Math.min(total - 1, Math.floor(total * 0.25))];
      if (fo >= top10FO) score += 1;
      if (fi <= bottom25FI) score += 1;
    }
    scores.push({ id, score, name: n.name, type: n.type, filePath: n.filePath || "" });
  }
  scores.sort((a, b) => b.score - a.score);
  const entryPointCandidates = scores.slice(0, 5).map((s) => ({
    id: s.id, score: s.score, name: s.name, summary: byId.get(s.id).summary || "",
  }));

  // ---- BFS from top CODE entry point (skip documents) ----
  // Prefer the dispatch-specified project entry point (src/exo/main.py) when present.
  const preferredEntry = "file:src/exo/main.py";
  const codeCandidates = entryPointCandidates.filter((c) => byId.get(c.id).type !== "document");
  const startNode =
    (nodeIds.has(preferredEntry) && byId.get(preferredEntry).type === "file")
      ? preferredEntry
      : (codeCandidates.length ? codeCandidates[0].id : entryPointCandidates[0].id);
  const order = [];
  const depthMap = {};
  const queue = [[startNode, 0]];
  const seen = new Set([startNode]);
  while (queue.length) {
    const [cur, depth] = queue.shift();
    order.push(cur);
    depthMap[cur] = depth;
    for (const next of fwd.get(cur) || []) {
      if (!seen.has(next)) {
        seen.add(next);
        queue.push([next, depth + 1]);
      }
    }
  }
  const byDepth = {};
  for (const [id, depth] of Object.entries(depthMap)) {
    (byDepth[depth] = byDepth[depth] || []).push(id);
  }
  const bfsTraversal = { startNode, order, depthMap, byDepth };

  // ---- Non-code file inventory ----
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const n of nodes) {
    const item = { id: n.id, name: n.name, type: n.type, summary: n.summary || "" };
    if (n.type === "document") nonCodeFiles.documentation.push(item);
    else if (["service", "pipeline", "resource"].includes(n.type)) nonCodeFiles.infrastructure.push(item);
    else if (["table", "schema", "endpoint"].includes(n.type)) nonCodeFiles.data.push(item);
    else if (n.type === "config") nonCodeFiles.config.push(item);
  }

  // ---- Tightly coupled clusters ----
  // bidirectional pairs: A<->B via imports/calls
  const pairKey = (a, b) => (a < b ? `${a}|||${b}` : `${b}|||${a}`);
  const clusters = [];
  const used = new Set();
  const pairs = new Map(); // pairKey -> [a, b]
  for (const e of edges) {
    const { source, target, type } = e;
    if (type !== "imports" && type !== "calls") continue;
    if (fwd.get(target) && fwd.get(target).has(source)) {
      // bidirectional
      pairs.set(pairKey(source, target), [source, target]);
    }
  }
  const seenPairs = new Set();
  for (const [key, [a, b]] of pairs.entries()) {
    if (a === b || seenPairs.has(key)) continue;
    seenPairs.add(key);
    const members = new Set([a, b]);
    // expand: add nodes with edges to 2+ members
    let changed = true;
    while (changed) {
      changed = false;
      for (const nid of nodeIds) {
        if (members.has(nid)) continue;
        let connects = 0;
        for (const m of members) {
          if (und.get(nid).has(m)) connects++;
        }
        if (connects >= 2 && members.size < 5) {
          members.add(nid);
          changed = true;
        }
      }
    }
    const list = [...members];
    if (list.length < 2) continue;
    let edgeCount = 0;
    for (const e of edges) {
      if (members.has(e.source) && members.has(e.target)) edgeCount++;
    }
    clusters.push({ nodes: list, edgeCount });
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount);
  const topClusters = clusters.slice(0, 10);

  // ---- Layers ----
  const layerList = layers.map((l) => ({
    id: l.id, name: l.name, description: l.description || "",
  }));

  // ---- Node summary index ----
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary || "" };
  }

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal,
    nonCodeFiles,
    clusters: topClusters,
    layers: { count: layerList.length, list: layerList },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
  };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  process.exit(0);
}

try {
  main();
} catch (err) {
  console.error("ERROR:", err && err.stack ? err.stack : err);
  process.exit(1);
}
