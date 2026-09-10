# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Constrained (guided) decoding — JSON-Schema output enforcement (T28).

A logits processor that masks sampling so the generated text must match a
JSON Schema (v1 subset). Hand-rolled, dependency-free byte-level matcher —
mirrors mlx-lm #1007's "slim internal pure-Python matcher" option, with the
interface shaped so xgrammar can be swapped in later.

Why byte-level: BPE pieces (Llama-3 byte-level, SentencePiece sentinels,
multi-byte UTF-8 straddling tokens) make char-level FSMs wrong. JSON
structure bytes are all single bytes, so a byte-level FSM is exact by
construction (multi-byte characters inside strings are handled since any
non-structural byte is legal there).

Interface: ``Callable[[mx.array, mx.array], mx.array]`` — the mlx_lm
logits-processor shape (tokens, logits) -> logits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

try:
    from exo.shared.logging import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger("exo.constrained")


# ---------------------------------------------------------------------------
# Tokenizer -> byte trie (the correctness-critical normalization)
# ---------------------------------------------------------------------------


def _is_sentence_piece(tokenizer: Any) -> bool:
    """Detect SentencePiece-style pieces (Ġ/Ċ sentinels) vs byte-level BPE.

    Byte-level BPE (Llama 3, Qwen3, tiktoken-style) uses a ``ByteLevel``
    decoder and stores raw bytes as latin-1 characters in the vocab. Real
    SentencePiece (Llama 2, Mistral, Qwen2.5) uses a unigram/BPE model with
    ``Ġ``/``Ċ`` sentinel characters.

    Discriminator: the backend decoder type. ``model.type`` is NOT reliable —
    the BPE model object has no ``.type`` instance attribute.
    """
    try:
        decoder_type = str(type(tokenizer.backend_tokenizer.decoder).__name__)
    except Exception:
        decoder_type = ""
    if decoder_type == "ByteLevel":
        return False
    try:
        model_type = str(type(tokenizer.backend_tokenizer.model).__name__).lower()
    except Exception:
        model_type = ""
    return model_type in {"unigram", "bpe", "wordpiece"}


def _piece_to_bytes(piece: str, is_sentence_piece: bool) -> bytes:
    """Normalize a tokenizer piece to its byte sequence.

    - Byte-level BPE (Llama 3, Qwen3): each char IS one byte (0x00-0xFF) —
      ``encode('latin-1')`` recovers the raw bytes.
    - SentencePiece (Llama 2, Mistral, Qwen2.5): sentinels Ġ=space, Ċ=newline,
      Ĉ/ĉ=tabs; replace then UTF-8 encode.
    """
    if is_sentence_piece:
        piece = (
            piece.replace("\u0120", " ")  # Ġ leading space
            .replace("\u010a", "\n")  # Ċ newline
            .replace("\u0108", "\t")  # Ĉ tab
            .replace("\u0109", "\t")  # ĉ tab (some tokenizers)
        )
        return piece.encode("utf-8")
    return piece.encode("latin-1", errors="replace")


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    token_id: int | None = None


def _build_vocab_trie(tokenizer: Any) -> tuple[_TrieNode, dict[int, bytes], int]:
    """Build a byte-trie of every vocab token.

    Returns (root, token_id_to_bytes, vocab_size). Building the 128k-token
    Llama-3 trie takes ~0.5-2s, so callers should cache it (see
    ``cached_vocab_trie``).
    """
    is_sp = _is_sentence_piece(tokenizer)
    root = _TrieNode()
    try:
        vocab: dict[str, int] = tokenizer.get_vocab()
    except Exception:
        vocab = {str(tokenizer.decode([i])): i for i in range(int(tokenizer.vocab_size))}
    id_to_bytes: dict[int, bytes] = {}
    for piece, token_id in vocab.items():
        piece_bytes = _piece_to_bytes(piece, is_sp)
        node = root
        for byte in piece_bytes:
            node = node.children.setdefault(byte, _TrieNode())
        node.token_id = token_id
        id_to_bytes[token_id] = piece_bytes
    return root, id_to_bytes, len(vocab)


def cached_vocab_trie(tokenizer: Any) -> tuple[_TrieNode, dict[int, bytes], int]:
    """Build + cache the vocab trie per tokenizer (module-level cache)."""
    try:
        name = str(tokenizer.name_or_path)
    except Exception:
        name = type(tokenizer).__name__
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    key = (name, vocab_size)
    if key not in _TRIE_CACHE:
        _TRIE_CACHE[key] = _build_vocab_trie(tokenizer)
    return _TRIE_CACHE[key]


_TRIE_CACHE: dict[tuple[str, int], tuple[_TrieNode, dict[int, bytes], int]] = {}


# ---------------------------------------------------------------------------
# JSON Schema (v1 subset) -> byte-level FSM
# ---------------------------------------------------------------------------

_OPEN_BRACE = ord("{")
_CLOSE_BRACE = ord("}")
_OPEN_BRACKET = ord("[")
_CLOSE_BRACKET = ord("]")
_COMMA = ord(",")
_COLON = ord(":")
_QUOTE = ord('"')
_BACKSLASH = ord("\\")

# JSON insignificant whitespace (RFC 8259): space, tab, LF, CR.
_WS_BYTES = (ord(" "), ord("\t"), ord("\n"), ord("\r"))

_SUPPORTED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "description",
    "additionalProperties",
}
_UNSUPPORTED_KEYWORDS = {
    "$ref",
    "$defs",
    "$schema",
    "oneOf",
    "anyOf",
    "allOf",
    "pattern",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "const",
    "not",
    "if",
    "then",
    "else",
}


class _FSM:
    """A byte-level finite state machine for a JSON fragment."""

    __slots__ = ("n_states", "transitions", "accept", "start")

    def __init__(self, n_states: int, transitions: list[dict[int, int]], accept: list[bool]) -> None:
        self.n_states = n_states
        self.transitions = transitions
        self.accept = accept
        self.start = 0


def _char_sequence(bytes_seq: list[int]) -> _FSM:
    """A linear chain over exact bytes; consuming the last byte reaches an
    explicit accepting state (standard n+1-state construction).

    The terminal state is separate from the byte-consumption states so that
    ``_sequence`` chaining (accept state -> next step's start transitions)
    and ``_allowed_mask``'s transition walk both work: every byte has an
    outgoing transition.
    """
    n = len(bytes_seq)
    transitions: list[dict[int, int]] = [{} for _ in range(n + 1)]
    accept = [False] * (n + 1)
    for i, b in enumerate(bytes_seq):
        transitions[i][b] = i + 1
    accept[n] = True
    return _FSM(n + 1, transitions, accept)


def _epsilon() -> _FSM:
    return _FSM(1, [{}], [True])


def _ws_tolerant(byte: int) -> _FSM:
    """A structural byte preceded by optional JSON whitespace.

    State 0: ws self-loops + ``byte`` -> 1; state 1: accept (with ws
    self-loops so trailing whitespace before the next structure is legal).
    Folding whitespace into the START state makes the zero-ws shortcut
    inherent — ``_sequence`` chaining from the previous accept adds both the
    ws bytes and the structural byte, so ``"a":`` and ``"a" :`` both work.
    """
    transitions: list[dict[int, int]] = [{}, {}]
    for b in _WS_BYTES:
        transitions[0][b] = 0
        transitions[1][b] = 1
    transitions[0][byte] = 1
    return _FSM(2, transitions, [False, True])


def _choice(branches: list[_FSM]) -> _FSM:
    """Union: try each branch from a common start state 0."""
    total = sum(b.n_states for b in branches) + 1
    transitions: list[dict[int, int]] = [{} for _ in range(total)]
    accept = [False] * total
    offset = 1
    for b in branches:
        for i in range(b.n_states):
            transitions[offset + i] = {bb: offset + t for bb, t in b.transitions[i].items()}
            accept[offset + i] = b.accept[i]
        for bb, t in b.transitions[0].items():
            transitions[0][bb] = offset + t
        offset += b.n_states
    return _FSM(total, transitions, accept)


def _sequence(steps: list[_FSM]) -> _FSM:
    """Concatenate FSMs, chaining accept states of earlier steps to the next.

    Only the LAST step's accept states are accept states of the composed FSM.
    Intermediate accept states exist solely to chain the next step's start
    transitions — exposing them would let a suffix begin (or, after ``_choice``
    / outer ``_sequence`` chaining, an outer continuation like ``}``) at a
    point where the sequence is not actually complete. (This leaked
    intermediate accepts caused JSON objects to close early, skipping
    required properties.)
    """
    total = sum(s.n_states for s in steps)
    transitions: list[dict[int, int]] = [{} for _ in range(total)]
    accept = [False] * total
    offset = 0
    for idx, s in enumerate(steps):
        last = idx == len(steps) - 1
        for i in range(s.n_states):
            transitions[offset + i] = {
                b: offset + t for b, t in s.transitions[i].items()
            }
            if last:
                accept[offset + i] = s.accept[i]
        offset += s.n_states
    offset = 0
    for idx, s in enumerate(steps[:-1]):
        next_fsm = steps[idx + 1]
        next_start_offset = offset + s.n_states
        for i in range(s.n_states):
            if s.accept[i]:
                for bb, t in next_fsm.transitions[0].items():
                    transitions[offset + i][bb] = next_start_offset + t
        offset += s.n_states
    return _FSM(total, transitions, accept)


def _star(inner: _FSM) -> _FSM:
    """Zero or more repetitions: accept-state loops back to inner start."""
    total = inner.n_states + 1
    transitions: list[dict[int, int]] = [{} for _ in range(total)]
    accept = [True] + list(inner.accept)
    for bb, t in inner.transitions[0].items():
        transitions[0][bb] = 1 + t
    for i in range(inner.n_states):
        transitions[1 + i] = {b: 1 + t for b, t in inner.transitions[i].items()}
    for i in range(inner.n_states):
        if inner.accept[i]:
            for bb, t in inner.transitions[0].items():
                transitions[1 + i][bb] = 1 + t
    return _FSM(total, transitions, accept)


def _string_fsm() -> _FSM:
    """A JSON string: " (escapes allowed) ... " — byte-level (handles multi-byte).

    Standard n+1 construction: state 0 is the start, state 1 is inside the
    string, 2 = after backslash, 3 = after \\u (4 hex digits), and state 4 is
    the DEDICATED accept state reached by the closing quote. The accept state
    has no outgoing transitions, so ``_sequence`` chaining decides what may
    follow (e.g. ``,`` or ``}`` after an object value) — never another quote.
    """
    # 0 = start (before opening quote), 1 = inside string, 2 = after backslash,
    # 3 = after \u (4 hex digits), 4 = accept (after closing quote)
    transitions: list[dict[int, int]] = [{} for _ in range(5)]
    accept = [False] * 5
    transitions[0][_QUOTE] = 1
    for b in range(256):
        if b == _QUOTE:
            transitions[1][b] = 4  # closing quote -> dedicated accept
            transitions[2][_QUOTE] = 1
        elif b == _BACKSLASH:
            transitions[1][b] = 2
        elif b < 0x20:
            pass  # raw control chars forbidden
        else:
            transitions[1][b] = 1
    for b in (ord('"'), ord("\\"), ord("/"), ord("b"), ord("f"), ord("n"), ord("r"), ord("t")):
        transitions[2][b] = 1
    transitions[2][ord("u")] = 3
    for b in range(256):
        if chr(b) in "0123456789abcdefABCDEF":
            transitions[3][b] = 1
    accept[4] = True
    return _FSM(5, transitions, accept)


def _number_fsm() -> _FSM:
    r"""-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"""
    digits = [ord(c) for c in "0123456789"]
    transitions: list[dict[int, int]] = [{} for _ in range(8)]
    accept = [False] * 8
    for b in digits:
        transitions[0][b] = 2 if b == ord("0") else 3
    transitions[0][ord("-")] = 1
    for b in digits:
        transitions[1][b] = 2 if b == ord("0") else 3
    accept[2] = True
    transitions[2][ord(".")] = 4
    transitions[2][ord("e")] = 6
    transitions[2][ord("E")] = 6
    accept[3] = True
    for b in digits:
        transitions[3][b] = 3
    transitions[3][ord(".")] = 4
    transitions[3][ord("e")] = 6
    transitions[3][ord("E")] = 6
    for b in digits:
        transitions[4][b] = 5
    accept[5] = True
    for b in digits:
        transitions[5][b] = 5
    transitions[5][ord("e")] = 6
    transitions[5][ord("E")] = 6
    transitions[6][ord("+")] = 7
    transitions[6][ord("-")] = 7
    for b in digits:
        transitions[6][b] = 7
    accept[7] = True
    for b in digits:
        transitions[7][b] = 7
    return _FSM(8, transitions, accept)


def _literal_fsm(text: str) -> _FSM:
    return _char_sequence([ord(c) for c in text])


def _escape_json_string(s: str) -> str:
    return json.dumps(s, ensure_ascii=False, separators=(",", ":"))


def _serialize_enum_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float, str)):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(v, separators=(",", ":"))


def _json_value_fsm(schema: dict[str, Any]) -> _FSM:
    """Compile a JSON Schema value (recursive) into a byte FSM."""
    if "enum" in schema:
        return _choice(
            [_char_sequence(list(_serialize_enum_value(v).encode("utf-8"))) for v in schema["enum"]]
        )
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return _choice([_json_value_fsm({"type": t}) for t in schema_type])
    if schema_type == "string":
        return _string_fsm()
    if schema_type in ("number", "integer"):
        return _number_fsm()
    if schema_type == "boolean":
        return _choice([_literal_fsm("true"), _literal_fsm("false")])
    if schema_type == "null":
        return _literal_fsm("null")
    if schema_type == "array":
        item_fsm = _json_value_fsm(schema.get("items", {"type": "string"}))
        open_b = _char_sequence([_OPEN_BRACKET])
        close_b = _char_sequence([_CLOSE_BRACKET])
        # [ ( item (, item)* )? ] — commas are ws-tolerant; item accepts
        # chain ',' and ']' (via the star's accepting start state).
        one_or_more = _sequence([item_fsm, _star(_sequence([_ws_tolerant(_COMMA), item_fsm]))])
        return _sequence([open_b, _choice([one_or_more, _epsilon()]), close_b])
    if schema_type == "object":
        return _object_fsm(schema)
    return _string_fsm()


def _object_fsm(schema: dict[str, Any]) -> _FSM:
    """Object: { "key" : value (, "key" : value )* }.

    Required keys come first. ``}`` (close) is legal only after ALL required
    keys have been emitted — optional keys may be skipped, so close is also
    legal after each optional key; it is never legal after a comma or while
    required keys remain. ``{}`` is only valid when no key is required.

    The chain is built manually (not with ``_choice`` over shared prefixes):
    ``_choice`` merges branch starts by overwriting duplicate byte keys, so
    branches sharing a required-key prefix (e.g. "close now" vs "continue with
    optionals") would make the earlier branch unreachable.
    """
    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])
    optional_keys = [k for k in properties if k not in required]

    open_brace = _char_sequence([_OPEN_BRACE])
    close_brace = _char_sequence([_CLOSE_BRACE])
    comma = _ws_tolerant(_COMMA)
    colon = _ws_tolerant(_COLON)

    def key_value(key: str, value_schema: dict[str, Any]) -> _FSM:
        return _sequence(
            [
                _char_sequence(list(_escape_json_string(key).encode("utf-8"))),
                colon,
                _json_value_fsm(value_schema),
            ]
        )

    required_kvs = [key_value(k, properties[k]) for k in required]
    optional_kvs = [key_value(k, properties[k]) for k in optional_keys]
    kvs = required_kvs + optional_kvs

    if not kvs:
        # { ws } — no properties at all.
        return _sequence([_char_sequence([_OPEN_BRACE]), _ws_tolerant(_CLOSE_BRACE)])

    # Layout: [open][kv0][comma][kv1][comma]...[kvN][close]
    pieces: list[_FSM] = [open_brace]
    for i, kv in enumerate(kvs):
        pieces.append(kv)
        if i < len(kvs) - 1:
            pieces.append(comma)
    pieces.append(close_brace)

    total = sum(p.n_states for p in pieces)
    transitions: list[dict[int, int]] = [{} for _ in range(total)]
    accept = [False] * total
    offsets: list[int] = []
    off = 0
    for p in pieces:
        for i in range(p.n_states):
            transitions[off + i] = {
                b: off + t for b, t in p.transitions[i].items()
            }
        offsets.append(off)
        off += p.n_states

    def piece_accepts(idx: int) -> list[int]:
        p = pieces[idx]
        return [offsets[idx] + s for s in range(p.n_states) if p.accept[s]]

    # open_brace accept -> kv0 start, with leading-ws self-loop.
    open_acc = offsets[0] + 1
    for bb in _WS_BYTES:
        transitions[open_acc][bb] = open_acc
    for bb, t in kvs[0].transitions[0].items():
        transitions[open_acc][bb] = offsets[1] + t
    # Empty object is valid only when nothing is required. The '}' must chain
    # through the close_brace's own transition so it lands on its ACCEPT
    # state — pointing at the start state would require a phantom second '}'.
    close_accept = offsets[-1] + 1
    if not required_kvs:
        transitions[open_acc][_CLOSE_BRACE] = close_accept  # "{}" with no ws

    last_required = len(required_kvs) - 1
    for i in range(len(kvs)):
        kv_idx = 1 + 2 * i
        kv_accs = piece_accepts(kv_idx)
        for a in kv_accs:
            # ws self-loop after the value (before ',' or '}').
            for bb in _WS_BYTES:
                transitions[a][bb] = a
            if i < len(kvs) - 1:
                # ',' -> the ws-tolerant comma's ACCEPT (its start state
                # merely owns the ws self-loops + the ',' transition; we
                # chain through that transition so one comma lands on the
                # accept — never ',,').
                comma_off = offsets[kv_idx + 1]
                transitions[a][_COMMA] = comma_off + 1
            # '}' closes once every required key has been emitted.
            if i >= last_required:
                transitions[a][_CLOSE_BRACE] = close_accept
        if i < len(kvs) - 1:
            # comma accept -> next kv start (comma accept has ws self-loops).
            comma_acc = offsets[kv_idx + 1] + 1
            next_kv_off = offsets[kv_idx + 2]
            for bb, t in kvs[i + 1].transitions[0].items():
                transitions[comma_acc][bb] = next_kv_off + t

    # The real terminal: the close_brace accept (drives EOS-when-complete).
    # Trailing insignificant whitespace keeps it complete (self-loop).
    accept[close_accept] = True
    for bb in _WS_BYTES:
        transitions[close_accept][bb] = close_accept
    return _FSM(total, transitions, accept)


def compile_json_schema(schema: dict[str, Any]) -> _FSM:
    """Validate + compile a JSON Schema (v1 subset) into a byte FSM.

    Raises ValueError with a clear message for unsupported keywords — the
    caller maps this to an HTTP 400 at request time (never mid-generation).
    """
    unsupported = [k for k in schema if k in _UNSUPPORTED_KEYWORDS]
    if unsupported:
        raise ValueError(
            f"Unsupported JSON Schema keyword(s) for constrained decoding: {sorted(unsupported)}. "
            f"Supported: {sorted(_SUPPORTED_KEYWORDS)}"
        )
    return _json_value_fsm(schema)


# ---------------------------------------------------------------------------
# The logits processor
# ---------------------------------------------------------------------------


class ConstrainedDecodingProcessor:
    """Masks logits so the output must match a JSON Schema.

    Implements the mlx_lm logits-processor interface:
    ``Callable[[mx.array, mx.array], mx.array]`` (tokens, logits) -> logits.

    FSM state is kept in Python and the latest token is force-evaluated
    (``int(tokens[-1].item())``) — see mlx-lm #1156 (stateful processors see
    stale tokens due to lazy evaluation).
    """

    def __init__(
        self,
        tokenizer: Any,
        schema: dict[str, Any],
        *,
        allow_eos_when_complete: bool = True,
        eos_token_id: int | None = None,
    ) -> None:
        self.schema = schema
        self.fsm = compile_json_schema(schema)
        self._trie_root, self._id_to_bytes, self._vocab_size = cached_vocab_trie(tokenizer)
        self._allow_eos_when_complete = allow_eos_when_complete
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.eos_token_id = eos_token_id
        self._state = self.fsm.start
        self._cache: dict[int, mx.array] = {}
        # Prompt-boundary tracking: the first invocation receives the prompt's
        # trailing tokens (mlx_lm passes the prompt context on the first
        # processor call, and BatchGenerator passes the full token history on
        # every call). We only walk tokens that are NEW since the previous
        # call, so generation starts from the schema's start state instead of
        # a state polluted by arbitrary prompt text.
        self._walked = 0
        # States where the generated output is COMPLETE — not merely
        # acceptably stoppable mid-structure. Sequence-built FSMs (objects,
        # arrays, booleans, numbers) end in an accepting terminal state
        # (n_states - 1); allowing EOS at INTERMEDIATE accept states (e.g.
        # right after "{", where the empty-object epsilon accept fires) makes
        # greedy sampling stop immediately with a partial/empty document.
        # Terminal = accept state with no outgoing transitions (or only
        # self-loops, e.g. trailing-ws on object/array close). The previous
        # 's == n_states - 1' heuristic broke for _choice FSMs where each
        # branch has its own accept at different offsets (boolean 'true'
        # terminates at state 5 of 12, not the last state), causing EOS to
        # be blocked and triggering the EXO_CD_006 deadlock fallback.
        self._complete_states = {
            s
            for s in range(self.fsm.n_states)
            if self.fsm.accept[s]
            and all(t == s for t in self.fsm.transitions[s].values())
        }
        # Last structural byte actually consumed from a generated token —
        # used to recognise the hand-built string FSM's "string just closed"
        # accept (state 0 reached via a closing quote) vs its start state.
        self._last_byte: int | None = None

    def reset(self) -> None:
        """Reset FSM state for a new request (keep the allowed-set cache)."""
        self._state = self.fsm.start
        self._walked = 0

    def _allowed_mask(self, state: int) -> mx.array:
        cached = self._cache.get(state)
        if cached is not None:
            return cached
        allowed: list[int] = []
        stack: list[tuple[_TrieNode, int]] = [(self._trie_root, state)]
        while stack:
            node, fsm_state = stack.pop()
            for byte, child in node.children.items():
                nxt = self.fsm.transitions[fsm_state].get(byte)
                if nxt is None:
                    continue
                if child.token_id is not None:
                    allowed.append(child.token_id)
                stack.append((child, nxt))
        if self.eos_token_id is not None:
            complete = self._state in self._complete_states
            # Hand-built string FSM: the accept is the START state reached
            # after a closing quote — complete only if the last byte emitted
            # was a quote (the start state at the very beginning must not
            # terminate).
            if (
                not complete
                and self.fsm.accept[self._state]
                and self._state == self.fsm.start
                and self._last_byte == _QUOTE
            ):
                complete = True
            if complete and self._allow_eos_when_complete:
                if self.eos_token_id not in allowed:
                    allowed.append(self.eos_token_id)
            elif self.eos_token_id in allowed:
                # EOS must NEVER leak into the allowed set as ordinary
                # content. Its token bytes are typically valid string content
                # (e.g. '<|im_end|>' is just '<', '|', ... bytes), so the
                # trie walk happily admits it mid-string; a greedy model
                # treats it as the easiest way to stop and ends the response
                # inside an unterminated string.
                allowed.remove(self.eos_token_id)
            if not allowed:
                # empty allowed set would deadlock — fall back to EOS
                logger.warning(
                    "Constrained decoding: no tokens allowed at FSM state %s — allowing EOS to avoid deadlock",
                    state,
                )
                allowed.append(self.eos_token_id)
        mask = mx.full((self._vocab_size,), -1e9, dtype=mx.float32)
        if allowed:
            # multiply(0) zeroes the allowed entries (mlx fork's ArrayAt has
            # no .set — only add/subtract/multiply/etc).
            mask = mask.at[mx.array(allowed, dtype=mx.int32)].multiply(0.0)
        self._cache[state] = mask
        return mask

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        n_tokens = tokens.shape[-1]
        if self._walked == 0 and n_tokens > 0:
            # First call. The batch path (GenerationBatch._step) appends the
            # first generated token to the history BEFORE invoking processors,
            # so the first call already contains it; the sequential path's
            # first call is prompt-only. Walk ONLY the last token:
            #   - batch path: that is the first generated token, which MUST
            #     advance the FSM (skipping it lets two mutually-exclusive
            #     continuations of the start state both be sampled — e.g.
            #     "{" then "{" producing "{{" — the T28 doubling bug);
            #   - sequential path: the final prompt token, which only advances
            #     if it is a genuine structural byte (e.g. a trailing '{' that
            #     the model should continue).
            self._walked = n_tokens
            last_id = int(tokens[-1].item())  # force eval (mlx-lm #1156)
            if last_id != self.eos_token_id:
                for byte in self._id_to_bytes.get(last_id, b""):
                    nxt = self.fsm.transitions[self._state].get(byte)
                    if nxt is None:
                        break
                    self._state = nxt
                    self._last_byte = byte
        elif n_tokens > self._walked:
            new_tokens = tokens[self._walked :]
            self._walked = n_tokens
            for i in range(new_tokens.shape[-1]):
                last_id = int(new_tokens[i].item())  # force eval (mlx-lm #1156)
                if last_id != self.eos_token_id:
                    for byte in self._id_to_bytes.get(last_id, b""):
                        nxt = self.fsm.transitions[self._state].get(byte)
                        if nxt is None:
                            break
                        self._state = nxt
                        self._last_byte = byte
        mask = self._allowed_mask(self._state)
        if mask.shape[-1] != logits.shape[-1]:
            # Vocab mismatch guard: the tokenizer's vocab (used to build the
            # mask) can differ from the model's output head (e.g. Qwen3.5-2B
            # has extra reserved tokens). Instead of failing open, resize the
            # mask so the constraint still applies to the overlapping ids:
            # - model vocab larger: pad with -inf (block unknown ids)
            # - model vocab smaller: truncate the mask
            if mask.shape[-1] < logits.shape[-1]:
                # pad with -inf (block unknown ids beyond the tokenizer vocab)
                mask = mx.concatenate(
                    [
                        mask,
                        mx.full(
                            (logits.shape[-1] - mask.shape[-1],),
                            -1e9,
                            dtype=mx.float32,
                        ),
                    ],
                    axis=-1,
                )
            else:
                mask = mask[..., : logits.shape[-1]]
        return logits + mask
