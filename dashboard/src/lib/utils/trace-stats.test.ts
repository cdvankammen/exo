// Unit tests for trace-stats helpers used by routes/traces/[taskId]/+page.svelte.
// F1 regression: outer span category with count 0 + subcategory events must
// NEVER produce Infinity/NaN in /step math or formatDuration rendering.
import { describe, it, expect } from "vitest";
import {
  parsePhases,
  formatDuration,
  formatPercentage,
  avgPerStep,
} from "./trace-stats";
import type {
  TraceCategoryStats,
  TraceStatsResponse,
} from "$lib/types/api";

// TraceStatsResponse shape subset used by the trace stats API
// (canonical type now imported from $lib/types/api).

function cat(totalUs: number, count: number): TraceCategoryStats {
  return { totalUs, count, minUs: 0, maxUs: totalUs, avgUs: count ? totalUs / count : 0 };
}

// Construct a TraceStatsResponse with outer category count 0 + subcategory events
// (the exact F1 scenario: cat "sync" count 0, cat "sync/compute" count N).
function f1Fixture(): TraceStatsResponse {
  const byCategory: Record<string, TraceCategoryStats> = {
    sync: cat(0, 0), // outer span, ZERO steps
    "sync/compute": cat(5_000_000, 2), // subcategory events exist
  };
  return {
    taskId: "trace-f1",
    totalWallTimeUs: 5_000_000,
    byCategory,
    byRank: {},
  };
}

describe("formatDuration", () => {
  it("returns — for non-finite values (Infinity/NaN)", () => {
    expect(formatDuration(Infinity)).toBe("—");
    expect(formatDuration(NaN)).toBe("—");
    expect(formatDuration(-Infinity)).toBe("—");
  });

  it("returns — for negative durations", () => {
    expect(formatDuration(-1)).toBe("—");
  });

  it("formats us/ms/s normally", () => {
    expect(formatDuration(500)).toBe("500us");
    expect(formatDuration(1500)).toBe("1.50ms");
    expect(formatDuration(1_500_000)).toBe("1.50s");
  });
});

describe("formatPercentage", () => {
  it("returns 0.0% when total is 0", () => {
    expect(formatPercentage(100, 0)).toBe("0.0%");
  });

  it("computes normally otherwise", () => {
    expect(formatPercentage(50, 100)).toBe("50.0%");
  });
});

describe("avgPerStep", () => {
  it("returns 0 when stepCount is 0 (never Infinity/NaN)", () => {
    expect(avgPerStep(5_000_000, 0)).toBe(0);
    expect(Number.isFinite(avgPerStep(5_000_000, 0))).toBe(true);
  });

  it("computes total/step when stepCount > 0", () => {
    expect(avgPerStep(5_000_000, 2)).toBe(2_500_000);
  });
});

describe("parsePhases (F1 regression)", () => {
  it("keeps a phase with outer span count 0 when subcategory events exist", () => {
    const phases = parsePhases(f1Fixture().byCategory);
    expect(phases).toHaveLength(1);
    expect(phases[0].name).toBe("sync");
    expect(phases[0].stepCount).toBe(0);
    expect(phases[0].subcategories).toHaveLength(1);
    expect(phases[0].subcategories[0].name).toBe("compute");
  });

  it("renders no Infinity/NaN for the F1 fixture (per-step + duration)", () => {
    const phases = parsePhases(f1Fixture().byCategory);
    const phase = phases[0];
    // Page math: normalizedTotal / normalizedStepCount with nodeCount=1.
    const perStep = avgPerStep(phase.totalUs, phase.stepCount);
    expect(Number.isFinite(perStep)).toBe(true);
    expect(perStep).toBe(0);
    // Subcategory /step (line 264/274 math) with outer count 0.
    const subPerStep = avgPerStep(phase.subcategories[0].stats.totalUs, phase.stepCount);
    expect(Number.isFinite(subPerStep)).toBe(true);
    expect(subPerStep).toBe(0);
    // formatDuration must never render "Infinity"/"NaN" strings.
    expect(formatDuration(perStep)).not.toMatch(/Infinity|NaN/);
    expect(formatDuration(subPerStep)).not.toMatch(/Infinity|NaN/);
    // By Rank variant (line 337 math): subcat.totalUs / phase.stepCount => guarded.
    const rankPerStep = avgPerStep(phase.subcategories[0].stats.totalUs, phase.stepCount);
    expect(Number.isFinite(rankPerStep)).toBe(true);
    // 0 is a valid finite duration and renders as 0us — the guard prevents
    // Infinity/NaN, not zero.
    expect(formatDuration(rankPerStep)).toBe("0us");
  });

  it("sorts subcategories and phases by totalUs desc", () => {
    const byCategory = {
      a: cat(100, 1),
      "a/small": cat(10, 1),
      "a/big": cat(90, 1),
      b: cat(200, 2),
      "b/only": cat(150, 1),
    };
    const phases = parsePhases(byCategory);
    expect(phases.map((p) => p.name)).toEqual(["b", "a"]);
    expect(phases[0].subcategories.map((s) => s.name)).toEqual(["only"]);
    expect(phases[1].subcategories.map((s) => s.name)).toEqual(["big", "small"]);
  });

  it("drops categories with no outer span", () => {
    const phases = parsePhases({ "subcat/only": cat(10, 1) });
    expect(phases).toHaveLength(0);
  });
});