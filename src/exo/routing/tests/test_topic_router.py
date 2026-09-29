"""Messages from the network are only parsed when something on this node receives them."""

from exo.routing.router import TopicRouter, malformed_event_log
from exo.routing.topics import PublishPolicy, TypedTopic
from exo.utils.channels import channel
from exo.utils.pydantic_ext import FrozenModel


class Ping(FrozenModel):
    n: int


PINGS = TypedTopic("pings", PublishPolicy.Always, Ping)


def make_router() -> TopicRouter[Ping]:
    networking_sender, _ = channel[tuple[str, bytes]]()
    return TopicRouter[Ping](PINGS, networking_sender)


async def test_message_is_delivered_to_receivers() -> None:
    router = make_router()
    send, recv = channel[Ping]()
    router.senders.add(send)

    await router.publish_bytes(PINGS.serialize(Ping(n=1)))

    assert recv.collect() == [Ping(n=1)]


async def test_message_nobody_receives_is_not_parsed() -> None:
    router = make_router()
    before = len(malformed_event_log())

    # Would fail validation if it were parsed, and would be logged as malformed
    await router.publish_bytes(b"not a ping")

    assert malformed_event_log()[before:] == []


async def test_parse_errors_still_surface_when_someone_receives() -> None:
    router = make_router()
    send, recv = channel[Ping]()
    router.senders.add(send)
    before = len(malformed_event_log())

    # The fork drops malformed events instead of raising, but it must still
    # notice them when a receiver is present.
    await router.publish_bytes(b"not a ping")

    assert recv.collect() == []
    recorded = malformed_event_log()[before:]
    assert len(recorded) == 1
    assert recorded[0]["topic"] == PINGS.topic
