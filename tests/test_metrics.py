from pathlib import Path

from downwell_ai.metrics import EpisodeMetrics
from downwell_ai.metrics import EpisodeTally
from downwell_ai.metrics import LearnMetrics
from downwell_ai.metrics import MetricsWriter
from downwell_ai.metrics import UpgradePick
from downwell_ai.metrics import plot_runs
from downwell_ai.metrics import read_metrics


def _info(*, is_hit: bool = False) -> dict:
    return {"reward_components": {"progress": 1.0}, "area": 1, "level": 2, "depth": 50.0, "is_hit": is_hit}


def _episode(step: int, *, is_evaluation: bool = False) -> EpisodeMetrics:
    tally = EpisodeTally()
    for is_hit in [False, True, True, True]:
        tally.add(1.0, _info(is_hit=is_hit))
    return tally.finish(step, _info(), is_terminated=True, eps=0.5, is_evaluation=is_evaluation)


def _write_run(directory: Path, episodes: int) -> Path:
    path = directory / "metrics.jsonl"
    with MetricsWriter(path) as writer:
        for step in range(1, episodes + 1):
            writer.write(LearnMetrics(step=step * 10, loss=1.0 / step, mean_q=step))
            writer.write(_episode(step * 10, is_evaluation=step % 5 == 0))
    return path


def test_episode_counts_hits():
    episode = _episode(10)

    assert episode.hits == {"total": 3}
    assert episode.components == {"progress": 4.0}


def test_episode_counts_side_rooms_shops_purchases_and_upgrade_picks():
    tally = EpisodeTally()
    pick = {"area": 1, "level": 1, "offered": ["apple", "drone", "candle"], "picked": "apple"}
    for events in [
        {"entered_side_room": True, "in_side_room": True},
        {"in_side_room": True, "entered_shop": True},
        {"in_side_room": True, "gems_spent": 150},
        {"in_side_room": True, "gems_spent": 50},
        {"upgrade": pick},
    ]:
        tally.add(0.0, {**_info(), **events})

    episode = tally.finish(5, _info(), is_terminated=True, eps=0.5)

    assert (episode.side_rooms, episode.side_room_steps, episode.shops) == (1, 4, 1)
    assert (episode.purchases, episode.gems_spent) == (2, 200)
    assert episode.upgrades == [UpgradePick(area=1, level=1, offered=["apple", "drone", "candle"], picked="apple")]


def test_metrics_read_back_what_was_written(tmp_path: Path):
    records = read_metrics(_write_run(tmp_path, episodes=3))

    assert records[1] == _episode(10)
    assert records[4] == LearnMetrics(step=30, loss=1.0 / 3, mean_q=3)


def test_reading_skips_records_from_an_older_format(tmp_path: Path):
    path = _write_run(tmp_path, episodes=1)
    with path.open("ab") as file:
        file.write(b'{"type":"episode","step":20,"epsilon":0.9}\n')

    assert len(read_metrics(path)) == 2


def test_episodes_written_before_eps_and_hits_existed_still_load(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(
        b'{"type":"episode","step":5,"episode_return":1.0,"episode_length":5,"area":1,"level":1,"depth":0.0,"is_terminated":true,"components":{}}\n'
    )

    (episode,) = read_metrics(path)

    assert isinstance(episode, EpisodeMetrics)
    assert episode.hits == {}
    assert episode.upgrades == []


def test_plots_one_run_and_a_comparison(tmp_path: Path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first = read_metrics(_write_run(tmp_path / "first", episodes=50))
    second = read_metrics(_write_run(tmp_path / "second", episodes=20))

    plot_runs({"first": first}, tmp_path / "plots.png")
    plot_runs({"first": first, "second": second}, tmp_path / "comparison.png")

    assert (tmp_path / "plots.png").stat().st_size > 0
    assert (tmp_path / "comparison.png").stat().st_size > 0
