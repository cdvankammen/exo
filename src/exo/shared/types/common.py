"""Core identity and common types.

:class:`Id` subclasses (:class:`NodeId`, :class:`ModelId`, :class:`CommandId`, :class:`SessionId`), :class:`Host`, and :class:`TruncatingString`."""

from typing import Any, Self
from uuid import uuid4

from pydantic import GetCoreSchemaHandler, field_validator
from pydantic_core import CoreSchema, core_schema

from exo.utils.pydantic_ext import FrozenModel


class Id(str):
    """Base identifier type: a UUID string with lazy default."""

    def __new__(cls, value: str | None = None) -> Self:
        return super().__new__(cls, value or str(uuid4()))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Use a plain string schema, wrapping through ``cls``."""
        # Just use a plain string schema
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


class NodeId(Id):
    """Identifier of a node in the swarm."""


class SystemId(Id):
    """Identifier of a system (process) in the swarm."""


class ModelId(Id):
    """Identifier of a model, with filesystem/URL-safe helpers.

    ``normalize`` maps ``/`` to ``--`` for use in paths; ``short`` returns
    the last path segment (e.g. ``org/model-name`` → ``model-name``).
    """

    def normalize(self) -> str:
        """Return the id with ``/`` replaced by ``--`` (filesystem-safe)."""
        return self.replace("/", "--")

    def short(self) -> str:
        """Return the final path segment of the model id."""
        return self.split("/")[-1]


class CommandId(Id):
    """Identifier of a command issued by the master."""


class TruncatingString(str):
    """String subclass whose repr truncates to ``truncate_length`` chars."""

    truncate_length: int = -1

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,  # pyright: ignore[reportAny]
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))

    def __repr__(self):
        """Truncate the repr to ``truncate_length`` characters."""
        tl = type(self).truncate_length
        return (
            f"<{type(self).__name__}: {self[:tl] + '...' if len(self) > tl else self}>"
        )


class SessionId(FrozenModel):
    """Session identity: master node plus election clock (Lamport timestamp)."""

    master_node_id: NodeId
    election_clock: int


class Host(FrozenModel):
    """Network host: ip address and port."""

    ip: str
    port: int

    def __str__(self) -> str:
        return f"{self.ip}:{self.port}"

    @field_validator("port")
    @classmethod
    def check_port(cls, v: int) -> int:
        """Reject out-of-range port numbers."""
        if not (0 <= v <= 65535):
            raise ValueError("Port must be between 0 and 65535")
        return v
