"""Ephemeral port selection.

``random_ephemeral_port()`` picks a safe OS ephemeral port for local listeners."""

import random


def random_ephemeral_port() -> int:
    port = random.randint(49153, 65535)
    return port - 1 if port <= 52415 else port
