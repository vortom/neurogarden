import pytest

from neurogarden.dojo.watch import main


def test_watch_runs_a_short_episode_and_prints_a_summary(capsys):
    assert main(["--brain", "random", "--max-steps", "5", "--tps", "0", "--ascii"]) == 0
    out = capsys.readouterr().out
    assert out.count("Day 1") == 5  # one frame per step
    assert "random: lived 5 ticks (0 days) | still alive" in out


def test_watch_rejects_unknown_brains():
    with pytest.raises(SystemExit):
        main(["--brain", "oracle"])
