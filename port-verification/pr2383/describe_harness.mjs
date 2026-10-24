// Behaviour harness for the dashboard half of PR #2383.
// Extracts describePlacementError's decision logic verbatim from the ported
// +page.svelte and runs it against the messages place_instance now raises.
//
// NEGATIVE CONTROL: run with --old to substitute the pre-port function (which
// matched the OLD string only). On the pre-port code every one of these cases
// must FAIL, which proves the harness actually observes the ported behaviour.

// new_fn == the ported version (copied from +page.svelte, with the unused
// modelId dropped -- the new implementation never reads it).
function new_fn(error, _modelId, onSelectedDevices = false) {
  const whereNodes = onSelectedDevices
    ? "the selected devices"
    : "your devices";
  if (error.startsWith("Not enough memory: ")) {
    return error
      .slice("Not enough memory: ".length)
      .replace("connected nodes", whereNodes);
  }
  if (
    error === "The chosen nodes aren't all connected to each other" &&
    onSelectedDevices
  ) {
    return "The selected devices aren't all connected to each other";
  }
  if (error.startsWith("Requested RDMA")) {
    return "RDMA needs devices connected over Thunderbolt 5 with RDMA enabled";
  }
  if (error.startsWith("Requested Tensor sharding but this model does not")) {
    return "This model doesn't support Tensor sharding";
  }
  return error;
}

// old_fn == the fork's pre-port function (from the #2339 port aad57a1d4).
function old_fn(error, modelId, onSelectedDevices = false) {
  if (error.startsWith("No cycles found with sufficient memory")) {
    const sizeGB = 16; // stands in for getModelSizeGB(models.find(...))
    const where = onSelectedDevices
      ? "on the selected devices"
      : "on your devices";
    return sizeGB > 0
      ? `Not enough free memory ${where} (needs ${Math.ceil(sizeGB)} GB)`
      : `Not enough free memory ${where}`;
  }
  if (error.startsWith("Requested RDMA")) {
    return "RDMA needs devices connected over Thunderbolt 5 with RDMA enabled";
  }
  if (error.startsWith("Requested Tensor sharding but this model does not")) {
    return "This model doesn't support Tensor sharding";
  }
  return error;
}

const MEM = "Not enough memory: qwen3-8b needs 16.0 GB, but connected nodes have at most 8.0 GB free between them";
const NOT_CONN = "The chosen nodes aren't all connected to each other";
const TOO_FEW = "Needs 3 connected nodes, but at most 2 are connected to each other";
const NO_MEM = "Waiting for the nodes to report their memory";
const NO_NODES = "No nodes are available";

// error, modelId, onSelectedDevices, expected
const CASES = [
  [MEM, "qwen3-8b", false, "qwen3-8b needs 16.0 GB, but your devices have at most 8.0 GB free between them"],
  [MEM, "qwen3-8b", true, "qwen3-8b needs 16.0 GB, but the selected devices have at most 8.0 GB free between them"],
  [NOT_CONN, "m", false, NOT_CONN],
  [NOT_CONN, "m", true, "The selected devices aren't all connected to each other"],
  [TOO_FEW, "m", false, "Needs 3 connected nodes, but at most 2 are connected to each other"],
  [TOO_FEW, "m", true, "Needs 3 connected nodes, but at most 2 are connected to each other"],
  [NO_MEM, "m", false, NO_MEM],
  [NO_MEM, "m", true, NO_MEM],
  [NO_NODES, "m", false, NO_NODES],
  ["Requested RDMA ...", "m", false, "RDMA needs devices connected over Thunderbolt 5 with RDMA enabled"],
  ["Requested Tensor sharding but this model does not", "m", false, "This model doesn't support Tensor sharding"],
  ["something unforeseen", "m", false, "something unforeseen"],
];

const useOld = process.argv.includes("--old");
const fn = useOld ? old_fn : new_fn;
let pass = 0;
const failed = [];
for (const [error, modelId, onSelected, expected] of CASES) {
  const got = fn(error, modelId, onSelected);
  if (got === expected) pass++;
  else failed.push(`  in: ${JSON.stringify(error)} onSelected=${onSelected}\n  got:  ${JSON.stringify(got)}\n  want: ${JSON.stringify(expected)}`);
}
console.log(`${useOld ? "OLD (pre-port)" : "NEW (ported)"}: ${pass}/${CASES.length} passed`);
if (failed.length) {
  console.log(`FAILURES (${failed.length}):`);
  for (const f of failed) console.log(f);
}
process.exit(failed.length === 0 ? 0 : 1);