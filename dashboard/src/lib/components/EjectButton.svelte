<script lang="ts">
  // NOTE: `fly`/`fade` MUST be imported. Svelte 5 compiles unimported
  // transition names into bare references (`()=>fly`), which throw a runtime
  // ReferenceError (`fly is not defined`) the instant the split-button tries
  // to animate — breaking the EJECT -> NO/YES flow introduced in ca8474c7.
  import { fade, fly } from "svelte/transition";

  interface Props {
    // NOTE: the parent renders <EjectButton {id} ... />, so this prop MUST be
    // named `id`. It used to be `instanceId`, which silently left the prop
    // undefined and made YES call `onConfirm(undefined)` -> DELETE
    // /instance/undefined -> 404 ("Failed to eject instance").
    id: string;
    onConfirm: (id: string) => void;
    // Model name, used to name the eject target instead of an opaque instance
    // id. Upstream PR #2324 applied the same idea to its confirm() string,
    // which this split-button replaced.
    modelName?: string;
  }

  let { id, onConfirm, modelName }: Props = $props();

  // "meta-llama/Llama-3.2-1B-Instruct-4bit" -> "Llama-3.2-1B-Instruct-4bit"
  const shortName = $derived(
    modelName?.split("/").pop() || modelName || id.slice(0, 8).toUpperCase(),
  );

  let confirming = $state(false);

  function stopAtCard(e: Event) {
    e.stopPropagation();
  }

  function handleEject(e: Event) {
    e.stopPropagation();
    confirming = true;
  }

  function handleNo(e: Event) {
    e.stopPropagation();
    confirming = false;
  }

  function handleYes(e: Event) {
    e.stopPropagation();
    confirming = false;
    onConfirm(id);
  }
</script>

<!--
  NOTE: this component is rendered INSIDE the instance card, and the card's own
  onclick/onkeydown select the card's model as the chat model. Without
  stopPropagation, clicking EJECT (or pressing Enter/Space on a focused EJECT /
  NO / YES button, which fires the card's key handler) selected the very model
  being ejected — the next message then silently relaunched it, which for a
  large model is a ~100 GB download. This mirrors upstream exo PR #2324, which
  fixed the same bug for the confirm()-based DELETE button this component
  replaced.
-->
<div class="inline-flex items-center overflow-hidden">
  {#if confirming}
    <button
      onclick={handleNo}
      onkeydown={stopAtCard}
      class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:border-red-500/50 hover:text-red-400 transition-all duration-200 cursor-pointer"
      transition:fly={{ x: -12, duration: 180 }}
      title={`Cancel eject of ${shortName}`}
    >
      NO
    </button>
    <button
      onclick={handleYes}
      onkeydown={stopAtCard}
      class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
      transition:fly={{ x: 12, duration: 180 }}
      title={`Eject the ${shortName} instance (${id.slice(0, 8).toUpperCase()})`}
    >
      YES
    </button>
  {:else}
    <button
      onclick={handleEject}
      onkeydown={stopAtCard}
      class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
      transition:fade={{ duration: 150 }}
      title={`Eject the ${shortName} instance (${id.slice(0, 8).toUpperCase()})`}
    >
      EJECT
    </button>
  {/if}
</div>
