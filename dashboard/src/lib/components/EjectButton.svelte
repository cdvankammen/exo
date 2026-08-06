<script lang="ts">
  interface Props {
    instanceId: string;
    onConfirm: (instanceId: string) => void;
  }

  let { instanceId, onConfirm }: Props = $props();

  let confirming = $state(false);

  function handleEject() {
    confirming = true;
  }

  function handleNo() {
    confirming = false;
  }

  function handleYes() {
    confirming = false;
    onConfirm(instanceId);
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
      title="Eject instance and delete weights"
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
