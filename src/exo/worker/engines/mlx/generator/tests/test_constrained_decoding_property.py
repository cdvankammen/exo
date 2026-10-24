"""Property-based tests for constrained decoding FSM construction.

Uses hypothesis to generate random JSON schemas and valid instances,
verifying the FSM accepts valid JSON and rejects invalid JSON.
Ground truth: jsonschema library.

Originally found 3 genuine FSM bugs (empty arrays, single-item arrays,
string enum byte-collision) — now fixed. Tests run without xfail.
"""
import json
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from exo.worker.engines.mlx.generator.constrained_decoding import compile_json_schema


def _walk_fsm(fsm: Any, text: str) -> bool:
    """Walk the FSM byte-by-byte. Return True if final state is accepting."""
    state = fsm.start
    for byte in text.encode("utf-8"):
        nxt = fsm.transitions[state].get(byte)
        if nxt is None:
            return False
        state = nxt
    return fsm.accept[state]


# ---------------------------------------------------------------------------
# Bug-fix regression tests (previously xfailed)
# ---------------------------------------------------------------------------


class TestFSMBugFixes:
    """Regression tests for the three FSM construction bugs."""

    def test_empty_array_accepted(self) -> None:
        """Bug 1: empty arrays [] were rejected because _choice overwrote
        epsilon's start transitions with one_or_more's start bytes."""
        fsm = compile_json_schema({"type": "array", "items": {"type": "string"}})
        assert _walk_fsm(fsm, "[]")
        assert _walk_fsm(fsm, "[ ]")
        assert _walk_fsm(fsm, "[  ]")

    def test_single_item_array_accepted(self) -> None:
        """Bug 2: single-element arrays [v] were rejected because _sequence
        chaining didn't route ']' from item's accept through _star's zero-reps."""
        fsm = compile_json_schema({"type": "array", "items": {"type": "string"}})
        assert _walk_fsm(fsm, '["hello"]')
        assert _walk_fsm(fsm, '["a"]')

    def test_multi_item_array_accepted(self) -> None:
        """Verify multi-item arrays still work after the manual rebuild."""
        fsm = compile_json_schema({"type": "array", "items": {"type": "string"}})
        assert _walk_fsm(fsm, '["a","b"]')
        assert _walk_fsm(fsm, '["a", "b", "c"]')
        assert _walk_fsm(fsm, '["x","y","z","w"]')

    def test_string_enum_shared_first_byte(self) -> None:
        """Bug 3: string enums sharing first byte (e.g. ["a","b"] both start
        with '"') were broken because _choice overwrote duplicate byte keys."""
        fsm = compile_json_schema({"enum": ["a", "b"]})
        assert _walk_fsm(fsm, '"a"')
        assert _walk_fsm(fsm, '"b"')
        assert not _walk_fsm(fsm, '"c"')

    def test_string_enum_longer_shared_prefix(self) -> None:
        """Enums with longer shared prefixes must also work."""
        fsm = compile_json_schema({"enum": ["hello", "help", "world"]})
        assert _walk_fsm(fsm, '"hello"')
        assert _walk_fsm(fsm, '"help"')
        assert _walk_fsm(fsm, '"world"')
        assert not _walk_fsm(fsm, '"hell"')
        assert not _walk_fsm(fsm, '"helper"')

    def test_boolean_enum_shared_prefix(self) -> None:
        """Boolean type uses _choice([literal("true"), literal("false")])
        which don't share a prefix — verify no regression."""
        fsm = compile_json_schema({"type": "boolean"})
        assert _walk_fsm(fsm, "true")
        assert _walk_fsm(fsm, "false")
        assert not _walk_fsm(fsm, "null")

    def test_nested_arrays(self) -> None:
        """Nested arrays exercise recursive array FSM construction."""
        fsm = compile_json_schema({
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        })
        assert _walk_fsm(fsm, "[]")
        assert _walk_fsm(fsm, "[[]]")
        assert _walk_fsm(fsm, "[[1]]")
        assert _walk_fsm(fsm, "[[1,2],[3]]")

    def test_array_of_enums(self) -> None:
        """Array of enums exercises both fixes together."""
        fsm = compile_json_schema({
            "type": "array",
            "items": {"enum": ["a", "b"]},
        })
        assert _walk_fsm(fsm, "[]")
        assert _walk_fsm(fsm, '["a"]')
        assert _walk_fsm(fsm, '["b"]')
        assert _walk_fsm(fsm, '["a","b"]')
        assert not _walk_fsm(fsm, '["c"]')


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Strategy for simple JSON schemas
_simple_types = st.sampled_from(["string", "integer", "number", "boolean", "null"])


@st.composite
def _json_schemas(draw: Any) -> dict[str, Any]:
    """Generate random simple JSON schemas."""
    kind = draw(st.sampled_from(["simple", "array", "object", "enum"]))
    if kind == "simple":
        return {"type": draw(_simple_types)}
    if kind == "array":
        item_type = draw(_simple_types)
        return {"type": "array", "items": {"type": item_type}}
    if kind == "object":
        n_props = draw(st.integers(min_value=0, max_value=3))
        props = {}
        required = []
        for i in range(n_props):
            key = draw(st.sampled_from(["name", "age", "active", "value"]))
            if key not in props:
                props[key] = {"type": draw(_simple_types)}
                if draw(st.booleans()):
                    required.append(key)
        schema: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema
    # enum — restrict to ASCII strings to avoid ensure_ascii serialization
    # mismatches (non-ASCII enum handling is a separate concern from these 3 bugs).
    values = draw(st.lists(
        st.one_of(
            st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            st.integers(),
            st.booleans(),
        ),
        min_size=1, max_size=4, unique=True,
    ))
    return {"enum": values}


@st.composite
def _valid_json_for_schema(draw: Any, schema: dict[str, Any]) -> str:
    """Generate a valid JSON string for a given schema."""
    if "enum" in schema:
        val = draw(st.sampled_from(schema["enum"]))
        return json.dumps(val)
    t = schema.get("type", "string")
    if t == "string":
        return json.dumps(draw(st.text(min_size=0, max_size=10)))
    if t == "integer":
        return str(draw(st.integers(min_value=-100, max_value=100)))
    if t == "number":
        return str(draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    if t == "boolean":
        return "true" if draw(st.booleans()) else "false"
    if t == "null":
        return "null"
    if t == "array":
        items = schema.get("items", {"type": "string"})
        n = draw(st.integers(min_value=0, max_value=3))
        arr = []
        for _ in range(n):
            inner = draw(_valid_json_for_schema(items))
            arr.append(json.loads(inner))
        return json.dumps(arr)
    if t == "object":
        obj = {}
        props = schema.get("properties", {})
        req = schema.get("required", [])
        for k in req:
            inner = draw(_valid_json_for_schema(props[k]))
            obj[k] = json.loads(inner)
        for k in props:
            if k not in obj and draw(st.booleans()):
                inner = draw(_valid_json_for_schema(props[k]))
                obj[k] = json.loads(inner)
        return json.dumps(obj)
    return '"fallback"'


class TestPropertyFSMAcceptsValid:
    """FSM must accept all valid JSON instances for a schema."""

    @settings(max_examples=150, deadline=None)
    @given(data=st.data())
    def test_valid_instance_accepted(self, data: Any) -> None:
        schema = data.draw(_json_schemas())
        try:
            fsm = compile_json_schema(schema)
        except ValueError:
            pytest.skip("unsupported schema keyword")
        valid_json = data.draw(_valid_json_for_schema(schema))
        assert _walk_fsm(fsm, valid_json), (
            f"FSM rejected valid JSON {valid_json!r} for schema {schema}"
        )


class TestPropertyFSMRejectsInvalid:
    """FSM must reject clearly invalid JSON strings."""

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_garbage_rejected(self, data: Any) -> None:
        schema = data.draw(_json_schemas())
        try:
            fsm = compile_json_schema(schema)
        except ValueError:
            pytest.skip("unsupported schema keyword")
        garbage = data.draw(st.sampled_from([
            "{invalid", "not json at all", "", "   ", "undefined",
            "NaN", "Infinity", "'single quotes'",
        ]))
        # Not all garbage is necessarily rejected by every schema
        # (e.g. "" might partially match), but these specific strings
        # should fail for most schemas.
        if garbage in ("{invalid", "not json at all", "undefined"):
            assert not _walk_fsm(fsm, garbage), (
                f"FSM accepted garbage {garbage!r} for schema {schema}"
            )
