#!/usr/bin/env node
// Structural analysis for architecture layer identification.
// Usage: node ua-arch-analyze.js <input.json> <output.json>
const fs = require('fs');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

let input;
try {
  input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to read input JSON:', e.message);
  process.exit(1);
}

const fileNodes = input.fileNodes || [];
const allEdges = input.allEdges || [];
const importEdges = allEdges.filter((e) => e.type === 'imports');

// ---- A. Directory grouping -------------------------------------------------
// Compute common path prefix shared by all file paths.
function filePathOf(node) {
  return node.filePath || (node.id || '').split(':').slice(1).join(':') || node.id;
}
const allPaths = fileNodes.map(filePathOf);
let commonPrefix = '';
if (allPaths.length > 0) {
  const first = allPaths[0];
  outer: for (let i = 0; i < first.length; i++) {
    const ch = first[i];
    for (const p of allPaths) {
      if (p[i] !== ch) break outer;
    }
    commonPrefix += ch;
  }
  // Trim common prefix to last '/'
  const lastSlash = commonPrefix.lastIndexOf('/');
  commonPrefix = lastSlash >= 0 ? commonPrefix.slice(0, lastSlash + 1) : '';
}

function groupOf(path) {
  const rel = path.startsWith(commonPrefix) ? path.slice(commonPrefix.length) : path;
  const segs = rel.split('/').filter(Boolean);
  if (segs.length === 0) return 'root';
  if (segs.length === 1) return '<root>';
  return segs[0];
}

const directoryGroups = {};
for (const n of fileNodes) {
  const g = groupOf(filePathOf(n));
  if (!directoryGroups[g]) directoryGroups[g] = [];
  directoryGroups[g].push(n.id);
}

// ---- B. Node type grouping -------------------------------------------------
const nodeTypeGroups = {};
for (const n of fileNodes) {
  const t = n.type || 'file';
  if (!nodeTypeGroups[t]) nodeTypeGroups[t] = [];
  nodeTypeGroups[t].push(n.id);
}

// ---- C. Import adjacency matrix --------------------------------------------
const fanIn = {};
const fanOut = {};
const importSet = {};
for (const n of fileNodes) {
  fanIn[n.id] = 0;
  fanOut[n.id] = 0;
  importSet[n.id] = new Set();
}
for (const e of importEdges) {
  if (fanOut[e.source] !== undefined) fanOut[e.source]++;
  if (fanIn[e.target] !== undefined) fanIn[e.target]++;
  if (importSet[e.source] !== undefined) importSet[e.source].add(e.target);
}

const groupOfId = (id) => {
  const node = fileNodes.find((n) => n.id === id);
  return node ? groupOf(filePathOf(node)) : 'unknown';
};

const groupImportsFrom = {}; // group -> Set of groups imported
const groupImportedBy = {}; // group -> Set of groups importing it
for (const g of Object.keys(directoryGroups)) {
  groupImportsFrom[g] = new Set();
  groupImportedBy[g] = new Set();
}
for (const e of importEdges) {
  const srcGroup = groupOfId(e.source);
  const tgtGroup = groupOfId(e.target);
  if (srcGroup !== tgtGroup && srcGroup !== 'unknown' && tgtGroup !== 'unknown') {
    groupImportsFrom[srcGroup].add(tgtGroup);
    groupImportedBy[tgtGroup].add(srcGroup);
  }
}

// ---- D. Cross-category dependency analysis ---------------------------------
const crossCategoryEdges = {};
for (const e of allEdges) {
  const srcNode = fileNodes.find((n) => n.id === e.source);
  const tgtNode = fileNodes.find((n) => n.id === e.target);
  if (!srcNode || !tgtNode) continue;
  const key = `${srcNode.type}->${tgtNode.type}::${e.type}`;
  if (!crossCategoryEdges[key]) crossCategoryEdges[key] = { count: 0 };
  crossCategoryEdges[key].count++;
}
const crossCategoryList = Object.entries(crossCategoryEdges).map(([k, v]) => {
  const [fromTo, edgeType] = k.split('::');
  const [fromType, toType] = fromTo.split('->');
  return { fromType, toType, edgeType, count: v.count };
});

// ---- E. Inter-group import frequency ---------------------------------------
const interGroupImports = {};
for (const e of importEdges) {
  const srcGroup = groupOfId(e.source);
  const tgtGroup = groupOfId(e.target);
  if (srcGroup === 'unknown' || tgtGroup === 'unknown') continue;
  const key = `${srcGroup}->${tgtGroup}`;
  if (!interGroupImports[key]) interGroupImports[key] = 0;
  interGroupImports[key]++;
}
const interGroupList = Object.entries(interGroupImports).map(([k, count]) => {
  const [from, to] = k.split('->');
  return { from, to, count };
});

// ---- F. Intra-group import density -----------------------------------------
const intraGroupDensity = {};
for (const g of Object.keys(directoryGroups)) {
  let internalEdges = 0;
  let totalEdges = 0;
  for (const e of importEdges) {
    const srcGroup = groupOfId(e.source);
    const tgtGroup = groupOfId(e.target);
    if (srcGroup === g || tgtGroup === g) {
      totalEdges++;
      if (srcGroup === tgtGroup) internalEdges++;
    }
  }
  intraGroupDensity[g] = {
    internalEdges,
    totalEdges,
    density: totalEdges > 0 ? +(internalEdges / totalEdges).toFixed(2) : 0,
  };
}

// ---- G. Directory pattern matching -----------------------------------------
const patternMap = {
  routes: 'api', api: 'api', controllers: 'api', endpoints: 'api', handlers: 'api', serializers: 'api', blueprints: 'api', routers: 'api',
  services: 'service', core: 'service', lib: 'service', domain: 'service', logic: 'service', composables: 'service', mailers: 'service', jobs: 'service', channels: 'service', signals: 'service', internal: 'service', worker: 'service', master: 'service', inference: 'service', download: 'service',
  models: 'data', db: 'data', data: 'data', persistence: 'data', repository: 'data', entities: 'data', migrations: 'data', entity: 'data', sql: 'data', database: 'data', schema: 'data',
  components: 'ui', views: 'ui', pages: 'ui', ui: 'ui', layouts: 'ui', screens: 'ui',
  middleware: 'middleware', plugins: 'middleware', interceptors: 'middleware', guards: 'middleware',
  utils: 'utility', helpers: 'utility', common: 'utility', shared: 'utility', tools: 'utility', templatetags: 'utility', pkg: 'utility',
  config: 'config', constants: 'config', env: 'config', settings: 'config', management: 'config', commands: 'config', resources: 'config', '.typings': 'config', '.mnemo': 'config', '.zed': 'config', '.feeder': 'config',
  test: 'test', tests: 'test', __tests__: 'test', spec: 'test', specs: 'test',
  types: 'types', interfaces: 'types', schemas: 'types', contracts: 'types', dtos: 'types', dto: 'types', request: 'types', response: 'types',
  hooks: 'hooks',
  store: 'state', state: 'state', reducers: 'state', actions: 'state', slices: 'state',
  assets: 'assets', static: 'assets', public: 'assets',
  cmd: 'entry', bin: 'entry',
  docs: 'documentation', documentation: 'documentation', wiki: 'documentation',
  deploy: 'infrastructure', deployment: 'infrastructure', infra: 'infrastructure', infrastructure: 'infrastructure', k8s: 'infrastructure', kubernetes: 'infrastructure', helm: 'infrastructure', charts: 'infrastructure', terraform: 'infrastructure', tf: 'infrastructure', docker: 'infrastructure',
  '.github': 'ci-cd', '.gitlab': 'ci-cd', '.circleci': 'ci-cd', '.githooks': 'ci-cd',
  rust: 'service', networking: 'service', exo_rs: 'service',
  dashboard: 'ui', src: 'ui', app: 'ui',
  bench: 'test', scripts: 'utility', tmp: 'utility',
  nix: 'infrastructure', packaging: 'infrastructure', python: 'config',
  shared: 'utility', routing: 'utility',
};

const patternMatches = {};
for (const g of Object.keys(directoryGroups)) {
  patternMatches[g] = patternMap[g] || null;
}

// File-level pattern matching
const filePatternMatches = {};
for (const n of fileNodes) {
  const name = n.name || filePathOf(n).split('/').pop();
  let match = null;
  if (/\.(test|spec)\.[^.]+$/.test(name) || /^test_.*\.py$/.test(name) || /_test\.(go|rs)$/.test(name)) match = 'test';
  else if (/\.d\.ts$/.test(name)) match = 'types';
  else if (name === 'index.ts' || name === 'index.js' || name === '__init__.py' || name === 'main.rs' || name === 'lib.rs') match = 'entry';
  else if (name === 'Cargo.toml' || name === 'go.mod' || name === 'Gemfile' || name === 'pom.xml' || name === 'build.gradle' || name === 'composer.json' || name === 'package.json' || name === 'pyproject.toml' || name === 'justfile') match = 'config';
  else if (/\.(md|rst)$/.test(name)) match = 'documentation';
  else if (/\.(tf|tfvars)$/.test(name)) match = 'infrastructure';
  else if (name === 'Dockerfile' || /^docker-compose.*/.test(name) || name === 'Makefile') match = 'infrastructure';
  else if (/\.sql$/.test(name)) match = 'data';
  else if (/\.(graphql|gql|proto)$/.test(name)) match = 'types';
  filePatternMatches[n.id] = match;
}

// ---- H. Deployment topology -------------------------------------------------
const infraNames = ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml'];
const infraFiles = [];
const isInfra = (p) =>
  /Dockerfile/.test(p) || /docker-compose/.test(p) || /\.(tf|tfvars)$/.test(p) || /(k8s|kubernetes|helm|charts)/.test(p) || /\.ya?ml$/.test(p) && /(deploy|infra|ci|workflow)/.test(p);
for (const n of fileNodes) {
  const p = filePathOf(n);
  if (isInfra(p)) infraFiles.push(p);
}
const deploymentTopology = {
  hasDockerfile: infraFiles.some((p) => /Dockerfile/.test(p)),
  hasCompose: infraFiles.some((p) => /docker-compose/.test(p)),
  hasK8s: infraFiles.some((p) => /(k8s|kubernetes|helm|charts)/.test(p)),
  hasTerraform: infraFiles.some((p) => /\.(tf|tfvars)$/.test(p)),
  hasCI: infraFiles.some((p) => /\.(github|gitlab|circleci)|workflow/.test(p)),
  infraFiles,
};

// ---- I. Data pipeline -------------------------------------------------------
const dataPipeline = {
  schemaFiles: fileNodes.filter((n) => /\.sql$/.test(filePathOf(n))).map((n) => n.id),
  migrationFiles: fileNodes.filter((n) => /migration/.test(filePathOf(n))).map((n) => n.id),
  dataModelFiles: fileNodes.filter((n) => /(models|schemas)/.test(filePathOf(n))).map((n) => n.id),
  apiHandlerFiles: fileNodes.filter((n) => /(routes|api|endpoints|handlers|controllers)/.test(filePathOf(n))).map((n) => n.id),
};

// ---- J. Documentation coverage ---------------------------------------------
const groupsWithDocs = new Set();
const docsByGroup = {};
for (const n of fileNodes) {
  if (n.type !== 'document') continue;
  const g = groupOf(filePathOf(n));
  if (!docsByGroup[g]) docsByGroup[g] = [];
  docsByGroup[g].push(n.id);
  groupsWithDocs.add(g);
}
const totalGroups = Object.keys(directoryGroups).length;
const docCoverage = {
  groupsWithDocs: groupsWithDocs.size,
  totalGroups,
  coverageRatio: totalGroups > 0 ? +(groupsWithDocs.size / totalGroups).toFixed(2) : 0,
  undocumentedGroups: Object.keys(directoryGroups).filter((g) => !groupsWithDocs.has(g)),
};

// ---- K. Dependency direction ------------------------------------------------
const dependencyDirection = [];
const pairCounts = {};
for (const { from, to, count } of interGroupList) {
  const key = [from, to].sort().join('<=>');
  if (!pairCounts[key]) pairCounts[key] = { a: from, b: to, aToB: 0, bToA: 0 };
  if (from < to || pairCounts[key].a === from) pairCounts[key].aToB += count;
  else pairCounts[key].bToA += count;
}
// recompute properly
const dirPairs = {};
for (const { from, to, count } of interGroupList) {
  const key = [from, to].sort().join('<=>');
  if (!dirPairs[key]) dirPairs[key] = { g1: null, g2: null, g1toG2: 0, g2toG1: 0 };
  const sorted = [from, to].sort();
  dirPairs[key].g1 = sorted[0];
  dirPairs[key].g2 = sorted[1];
  if (from === sorted[0]) dirPairs[key].g1toG2 += count;
  else dirPairs[key].g2toG1 += count;
}
for (const v of Object.values(dirPairs)) {
  if (v.g1toG2 > v.g2toG1) dependencyDirection.push({ dependent: v.g1, dependsOn: v.g2 });
  else if (v.g2toG1 > v.g1toG2) dependencyDirection.push({ dependent: v.g2, dependsOn: v.g1 });
}

// ---- Output ----------------------------------------------------------------
const result = {
  scriptCompleted: true,
  commonPrefix,
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges: crossCategoryList,
  interGroupImports: interGroupList,
  intraGroupDensity,
  patternMatches,
  filePatternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats: {
    totalFileNodes: fileNodes.length,
    filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([g, ids]) => [g, ids.length])),
    nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([t, ids]) => [t, ids.length])),
  },
  fileFanIn: Object.fromEntries(Object.entries(fanIn).sort((a, b) => b[1] - a[1]).slice(0, 30)),
  fileFanOut: Object.fromEntries(Object.entries(fanOut).sort((a, b) => b[1] - a[1]).slice(0, 30)),
};

fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
console.log(`Analysis complete. ${fileNodes.length} file nodes, ${allEdges.length} edges.`);
process.exit(0);
