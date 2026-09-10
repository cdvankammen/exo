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
  }

  let { id, onConfirm }: Props = $props();

  let confirming = $state(false);

  function handleEject() {
    confirming = true;
  }

  function handleNo() {
    confirming = false;
  }

  function handleYes() {
    confirming = false;
    onConfirm(id);
  }
</script>

<div class="inline-flex items-center overflow-hidden">
  {#if confirming}
    <button
      onclick={handleNo}
      class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:border-red-500/50 hover:text-red-400 transition-all duration-200 cursor-pointer"
      transition:fly={{ x: -12, duration: 180 }}
      title="Cancel eject"
    >
      NO
    </button>
    <button
      onclick={handleYes}
      class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
      transition:fly={{ x: 12, duration: 180 }}
      title="Eject instance"
    >
      YES
    </button>
  {:else}
    <button
      onclick={handleEject}
      class="text-xs px-2 py-1 font-mono tracking-wider uppercase border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/50 transition-all duration-200 cursor-pointer"
      transition:fade={{ duration: 150 }}
    >
      EJECT
    </button>
  {/if}
</div>
