import asyncio

import numpy as np

from neurogarden.engine import World, maps
from neurogarden.protocol import build_catalog
from neurogarden.protocol.codec import decode_channels, encode_channels
from neurogarden.protocol.mailbox import Mailbox
from neurogarden.protocol.messages import (
    Chronicle,
    ChronicleMessage,
    Joined,
    JoinedMessage,
    Observation,
    ObservationMessage,
)


def observation(tick):
    world = World.from_map(maps.load("drosoville"))
    fly = world.spawn()
    payload = Observation(
        tick=tick,
        deadline_ms=200,
        channels=encode_channels(world.observe(fly)),
        events=[],
        missed=0,
    )
    return ObservationMessage(payload=payload)


def test_latest_observation_replaces_an_undelivered_one_but_keeps_order_of_the_rest():
    async def scenario():
        box = Mailbox()
        joined = JoinedMessage(
            payload=Joined(agent_id=1, lineage=1, name="Dusty Wing", tick=0, reattached=False)
        )
        box.put(joined)
        box.put(observation(0))
        box.put(ChronicleMessage(payload=Chronicle(tick=0, text="hi")))
        box.put(observation(1))
        box.put(observation(2))
        assert len(box) == 3
        return [await box.get() for _ in range(3)]

    joined, note, latest = asyncio.run(scenario())
    assert joined.type == "joined"
    assert note.type == "chronicle"
    assert latest.type == "observation" and latest.payload.tick == 2


def test_get_waits_for_a_put_and_close_ends_the_stream():
    async def scenario():
        box = Mailbox()

        async def later():
            await asyncio.sleep(0.01)
            box.put(observation(7))
            box.close()

        task = asyncio.create_task(later())
        first = await box.get()
        second = await box.get()
        await task
        return first, second

    first, second = asyncio.run(scenario())
    assert first.payload.tick == 7 and second is None


def test_close_with_an_error_raises_it_once_drained():
    async def scenario():
        box = Mailbox()
        box.put(observation(1))
        box.close(RuntimeError("gone"))
        first = await box.get()
        try:
            await box.get()
        except RuntimeError as err:
            return first, str(err)

    first, error = asyncio.run(scenario())
    assert first.payload.tick == 1 and error == "gone"


def test_codec_round_trips_the_dojo_observation_exactly():
    world = World.from_map(maps.load("drosoville"), seed=3)
    fly = world.spawn()
    world.step({fly: 1})
    original = world.observe(fly)
    body = build_catalog().bodies["fly"]
    decoded = decode_channels(encode_channels(original), body)
    assert set(decoded) == set(original)
    for name in original:
        assert decoded[name].dtype == original[name].dtype, name
        assert decoded[name].shape == original[name].shape, name
        assert np.array_equal(decoded[name], original[name]), name
