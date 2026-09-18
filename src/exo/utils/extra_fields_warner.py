"""Strict-extra mixin for public API request models.

Pydantic v2 defaults to ``extra="ignore"`` for ``BaseModel``, which silently
drops unknown fields. For internal data structures this is fine, but for
public API request bodies it makes mistyped or out-of-spec fields invisible:
the request is accepted, the offending field is dropped, and the request
runs with whatever defaults the matching declared fields had.

In practice this has bitten us at the placement layer (``/place_instance``):
a mistyped ``instance_meta`` field was silently dropped, falling back to the
default ``MlxRing`` instead of the requested ``MlxJaccl``, and the only
symptom was lower decode throughput. No warning, no log line, no 4xx.

That problem was first surfaced by :class:`WarnExtraModel` (log-only
deprecation path, shipped 2026-08-08, exo-explore/exo#2015). After a
minor-version window with no reported breakage, the follow-up
(exo-explore/exo#2313) flipped the public API request models to
``extra="forbid"``: unknown fields now raise a ``ValidationError`` instead of
being silently dropped.

This module exposes :class:`StrictExtraModel`, the current base class for
public API request models. Unlike the old warn-only base, unknown fields are
rejected outright — that is the point of the flip. Subclasses that need to
evolve their request shape in a non-breaking way should add the new field to
the model and ship it in the same release as the clients that send it.

NOTE: models that declare ``frozen=True`` cannot inherit from this base
(pydantic v2.12: "a frozen class cannot inherit from a class that is not
frozen"). Apply this only to non-frozen request models.
"""

from pydantic import BaseModel, ConfigDict


class StrictExtraModel(BaseModel):
    """Base class for public API request models that reject unknown fields.

    Subclasses set ``extra="forbid"``: any key not declared on the model is a
    ``ValidationError``. This is the enforcement half of the deprecation path
    started by ``WarnExtraModel`` (exo-explore/exo#2015 → #2313).
    """

    model_config = ConfigDict(extra="forbid")


# Backward-compatible alias: the deprecation-path name still imports for any
# client that referenced it directly. New code should use StrictExtraModel.
WarnExtraModel = StrictExtraModel