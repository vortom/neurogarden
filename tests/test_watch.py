import functools

import pytest

from neurogarden.brains import RandomBrain
from neurogarden.dojo.env import NeuroGardenEnv
from neurogarden.dojo.watch import main
from neurogarden.engine.config import Config


def test_watch_runs_a_short_episode_and_prints_a_summary(capsys):
    assert main(["--brain", "random", "--max-steps", "5", "--tps", "0", "--ascii"]) == 0
    out = capsys.readouterr().out
    assert out.count("Day 1") == 5  # one frame per step
    assert "random: lived 5 ticks (0 days) | still alive" in out


def test_watch_rejects_unknown_brains():
    with pytest.raises(SystemExit):
        main(["--brain", "oracle"])


def test_watch_renders_the_death_frame(monkeypatch, capsys):
    dying_config = Config(initial_satiety=1, initial_health=5)
    monkeypatch.setattr(
        "neurogarden.dojo.watch.NeuroGardenEnv",
        functools.partial(NeuroGardenEnv, config=dying_config),
    )
    assert main(["--brain", "random", "--max-steps", "5", "--tps", "0", "--ascii"]) == 0
    out = capsys.readouterr().out
    assert "the fly has died" in out


def test_watch_handles_a_keyboard_interrupt_and_still_prints_the_summary(monkeypatch, capsys):
    calls = {"count": 0}
    original_act = RandomBrain.act

    def flaky_act(self, observation):
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt
        return original_act(self, observation)

    monkeypatch.setattr(RandomBrain, "act", flaky_act)
    assert main(["--brain", "random", "--max-steps", "50", "--tps", "0", "--ascii"]) == 0
    out = capsys.readouterr().out
    assert "random: lived" in out
