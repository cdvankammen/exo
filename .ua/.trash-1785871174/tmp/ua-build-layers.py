#!/usr/bin/env python3
"""Assign every file-level node to exactly one architecture layer and write layers.json."""
import json
from collections import Counter, defaultdict

INPUT = '.ua/tmp/ua-arch-input.json'
OUTPUT = '.ua/intermediate/layers.json'

with open(INPUT) as f:
    data = json.load(f)
nodes = data['fileNodes']

def pathof(n):
    return n.get('filePath') or n.get('id').split(':', 1)[1]

def is_test_path(p):
    segs = p.split('/')
    return ('tests' in segs) or (segs[0] == 'tests')

def assign(n):
    p = pathof(n)
    if is_test_path(p):
        return 'layer:test'
    if p.startswith('src/exo/api/'):
        return 'layer:api'
    if p.startswith('src/exo/master/'):
        return 'layer:control-plane'
    if p.startswith('src/exo/download/'):
        return 'layer:control-plane'
    if p.startswith('src/exo/routing/'):
        return 'layer:control-plane'
    if p.startswith('src/exo/worker/'):
        return 'layer:worker'
    if p.startswith('src/exo/shared/'):
        return 'layer:foundation'
    if p.startswith('src/exo/utils/'):
        return 'layer:foundation'
    if p.startswith('src/exo/'):
        return 'layer:foundation'  # top-level package files (__init__, __main__, main)
    if p.startswith('dashboard/'):
        return 'layer:client-apps'
    if p.startswith('app/'):
        return 'layer:client-apps'
    if p.startswith('rust/'):
        return 'layer:rust-core'
    if p.startswith('resources/'):
        return 'layer:model-data'
    if p.startswith('.typings/'):
        return 'layer:type-stubs'
    return 'layer:infrastructure'

layers = {
    'layer:api': {
        'id': 'layer:api',
        'name': 'API Layer',
        'description': 'FastAPI HTTP server exposing OpenAI-compatible chat completions and image generation, with request/response adapters for Claude, Ollama, and OpenAI Responses.',
    },
    'layer:control-plane': {
        'id': 'layer:control-plane',
        'name': 'Cluster Control Plane',
        'description': 'Cluster coordination: master election and model placement, zenoh pub/sub event routing, and the distributed model download coordinator.',
    },
    'layer:worker': {
        'id': 'layer:worker',
        'name': 'Inference Worker Layer',
        'description': 'Inference workers: MLX and image engine builders, runner supervision, disaggregated serving, and shard management on each node.',
    },
    'layer:foundation': {
        'id': 'layer:foundation',
        'name': 'Shared Foundation & Utilities',
        'description': 'Shared foundation: event-sourcing types and cluster state models, model card schemas, constants, topology, election protocol, and cross-cutting Python utilities.',
    },
    'layer:test': {
        'id': 'layer:test',
        'name': 'Test Suite',
        'description': 'Unit tests co-located under module test directories plus top-level multi-node integration tests, fixtures, and harnesses.',
    },
    'layer:client-apps': {
        'id': 'layer:client-apps',
        'name': 'Client Applications',
        'description': 'Client applications: the Svelte 5 + TypeScript dashboard frontend and the native Swift EXO menu-bar app for iOS/macOS.',
    },
    'layer:rust-core': {
        'id': 'layer:rust-core',
        'name': 'Rust Networking Core',
        'description': 'Rust components: zenoh networking primitives, PyO3 exo_rs bindings exposing networking to Python, and system custodian operations.',
    },
    'layer:model-data': {
        'id': 'layer:model-data',
        'name': 'Model Card Data',
        'description': 'TOML model card definitions declaring supported inference and image-generation models with their shard layouts and quantization variants.',
    },
    'layer:type-stubs': {
        'id': 'layer:type-stubs',
        'name': 'Vendored Type Stubs',
        'description': 'Vendored .pyi type stubs for third-party MLX, mflux, pynvml, and safetensors dependencies used to satisfy strict type checking.',
    },
    'layer:infrastructure': {
        'id': 'layer:infrastructure',
        'name': 'Infrastructure, Config & Tooling',
        'description': 'Build and CI/CD configuration, nix packaging, DMG/pyinstaller distribution, benchmarks, helper scripts, editor/agent configs, and project documentation.',
    },
}

assignments = defaultdict(list)
unassigned = []
for n in nodes:
    lid = assign(n)
    assignments[lid].append(n['id'])

# Verification
total = sum(len(v) for v in assignments.values())
ids_seen = set()
dupes = []
for lid, ids in assignments.items():
    for i in ids:
        if i in ids_seen:
            dupes.append(i)
        ids_seen.add(i)

print(f'Total nodes: {total} (expected {len(nodes)})')
print(f'Duplicates: {len(dupes)}')
if len(ids_seen) != len(nodes):
    missing = set(n["id"] for n in nodes) - ids_seen
    print(f'MISSING: {len(missing)}')
    for m in sorted(missing):
        print('  ', m)

missing_ids = set(n['id'] for n in nodes) - ids_seen
if len(ids_seen) != len(nodes) or dupes:
    raise SystemExit(1)

# Build output in defined layer order
order = ['layer:api', 'layer:control-plane', 'layer:worker', 'layer:foundation',
         'layer:test', 'layer:client-apps', 'layer:rust-core',
         'layer:model-data', 'layer:type-stubs', 'layer:infrastructure']
output = []
for lid in order:
    node_ids = sorted(assignments[lid])
    layer = dict(layers[lid])
    layer['nodeIds'] = node_ids
    output.append(layer)

with open(OUTPUT, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nWrote {OUTPUT}')
print('--- Layer summary ---')
for layer in output:
    print(f"{layer['id']}: {len(layer['nodeIds'])} nodes")
