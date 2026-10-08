<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    title,
    subtitle,
    toolbar,
    footer,
    children,
    class: className = '',
  }: {
    title: string;
    subtitle?: string;
    toolbar?: Snippet; // header-row right side (buttons, badges)
    footer?: Snippet;  // bottom block (config pre, actions)
    children: Snippet; // panel body
    class?: string;
  } = $props();
</script>

<div class="border border-exo-light-gray/20 rounded-lg bg-exo-medium-gray/20 overflow-hidden {className}">
  <!-- title row: same anatomy as IntegrationCard:38 + panel headers.
       flex-col (not row): IntegrationCard stacks title over subtitle; a row
       would move the subtitle beside the title and break visual parity (W2). -->
  <div class="flex items-center justify-between px-5 py-4">
    <div class="flex flex-col min-w-0">
      <h3 class="text-white text-sm font-semibold tracking-wide truncate">{title}</h3>
      {#if subtitle}<p class="text-exo-light-gray/60 text-xs mt-0.5 font-mono truncate">{subtitle}</p>{/if}
    </div>
    {#if toolbar}{@render toolbar()}{/if}
  </div>
  {@render children()}
  {#if footer}
    <div class="border-t border-exo-light-gray/10">{@render footer()}</div>
  {/if}
</div>