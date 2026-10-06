// Pure trace-stats helpers for the trace detail page
// (routes/traces/[taskId]/+page.svelte). Extracted from the page's instance
// script so the phase/step math is unit-testable (see trace-stats.test.ts).
//
// F1 guard: when an outer span category has count 0 but subcategory events
// exist (e.g. cat "sync" count 0 with cat "sync/compute" count N), any
// `total / stepCount` computation produces Infinity/NaN. avgPerStep() clamps
// stepCount 0 -> 0 and formatDuration() renders non-finite/negative as "—".

import type { TraceCategoryStats } from "$lib/types/api";

export type PhaseData = {
  name: string;
  subcategories: { name: string; stats: TraceCategoryStats }[];
  totalUs: number; // From outer span (e.g., "sync" category)
  stepCount: number; // Count of outer span events
};

/** Format a duration in microseconds; non-finite/negative -> "—". */
export function formatDuration(us: number): string {
  if (!Number.isFinite(us) || us < 0) return "—";
  if (us < 1000) return `${us.toFixed(0)}us`;
  if (us < 1_000_000) return `${(us / 1000).toFixed(2)}ms`;
  return `${(us / 1_000_000).toFixed(2)}s`;
}

/** Percentage of part/total; total 0 -> "0.0%". */
export function formatPercentage(part: number, total: number): string {
  if (total === 0) return "0.0%";
  return `${((part / total) * 100).toFixed(1)}%`;
}

/**
 * Guarded average: stepCount 0 -> 0 (never Infinity/NaN).
 * Use for every "/step" computation.
 */
export function avgPerStep(totalUs: number, stepCount: number): number {
  return stepCount === 0 ? 0 : totalUs / stepCount;
}

/**
 * Parse hierarchical categories like "sync/compute" into phases.
 * Only phases with an outer span (category without "/") are kept; the outer
 * span's totalUs/count define the phase row. A phase can legitimately have
 * stepCount 0 when subcategory events exist without any outer-span event.
 */
export function parsePhases(
  byCategory: Record<string, TraceCategoryStats>,
): PhaseData[] {
  const phases = new Map<
    string,
    {
      subcats: Map<string, TraceCategoryStats>;
      outerStats: TraceCategoryStats | null;
    }
  >();

  for (const [category, catStats] of Object.entries(byCategory)) {
    if (category.includes("/")) {
      const [phase, subcat] = category.split("/", 2);
      if (!phases.has(phase)) {
        phases.set(phase, { subcats: new Map(), outerStats: null });
      }
      phases.get(phase)!.subcats.set(subcat, catStats);
    } else {
      // Outer span - this IS the phase total
      if (!phases.has(category)) {
        phases.set(category, { subcats: new Map(), outerStats: null });
      }
      phases.get(category)!.outerStats = catStats;
    }
  }

  return Array.from(phases.entries())
    .filter(([_, data]) => data.outerStats !== null) // Only phases with outer spans
    .map(([name, data]) => ({
      name,
      subcategories: Array.from(data.subcats.entries())
        .map(([subName, subStats]) => ({ name: subName, stats: subStats }))
        .sort((a, b) => b.stats.totalUs - a.stats.totalUs),
      totalUs: data.outerStats!.totalUs, // Outer span total
      stepCount: data.outerStats!.count, // Number of steps
    }))
    .sort((a, b) => b.totalUs - a.totalUs);
}