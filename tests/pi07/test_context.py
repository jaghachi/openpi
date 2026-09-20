import numpy as np
import pytest

from openpi.pi07.context import DropoutConfig
from openpi.pi07.context import EpisodeMetadata
from openpi.pi07.context import PromptContext
from openpi.pi07.context import bin_episode_steps
from openpi.pi07.context import runtime_metadata
from openpi.pi07.context import sample_context_dropout


def test_prompt_and_numpy_scalars():
    false_label = False
    context = PromptContext(
        np.asarray("peel vegetables"),
        subtask="pick up the peeler.",
        metadata=EpisodeMetadata(np.int64(7999), np.asarray(5), mistake=np.asarray(false_label)),
        control_mode=np.asarray("joint"),
    )
    assert context.to_text() == (
        "Task: peel vegetables. Subtask: pick up the peeler. Speed: 8000. "
        "Quality: 5. Mistake: false. Control Mode: joint."
    )


@pytest.mark.parametrize(
    ("value", "expected"), [(0, 0), (249, 0), (250, 500), (1749, 1500), (1750, 2000), (2249, 2000), (2250, 2500)]
)
def test_speed_bins_use_half_up(value, expected):
    assert bin_episode_steps(value) == expected


@pytest.mark.parametrize("value", [True, "2000", np.array([2000])])
def test_speed_bins_reject_non_numeric_scalars(value):
    with pytest.raises(TypeError):
        bin_episode_steps(value)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_speed_bins_reject_invalid_numbers(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        bin_episode_steps(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quality", 4.9),
        ("quality", 5.0),
        ("quality", True),
        ("quality", "5"),
        ("speed", 20.1),
        ("speed", "2000"),
        ("speed", True),
        ("mistake", "false"),
        ("mistake", 0),
        ("mistake", np.array([False])),
    ],
)
def test_metadata_never_coerces_bad_labels(field, value):
    with pytest.raises(TypeError):
        EpisodeMetadata(**{field: value})


@pytest.mark.parametrize("quality", [0, 6])
def test_quality_range(quality):
    with pytest.raises(ValueError, match="Quality"):
        EpisodeMetadata(quality=quality)


def test_runtime_defaults_use_per_task_fifteenth_percentile():
    metadata = runtime_metadata([1000, 2000, 3000, 4000, 5000])
    assert metadata == EpisodeMetadata(1500, 5, mistake=False)


@pytest.mark.parametrize("lengths", [[], [1000, 0], [1000, True], [1000, 1500.2]])
def test_runtime_defaults_reject_bad_durations(lengths):
    with pytest.raises((ValueError, TypeError)):
        runtime_metadata(lengths)


def test_optional_memory_and_controls():
    context = PromptContext("clean", language_memory=("open fridge", "get food"), control_mode="ee")
    assert context.to_text() == "Task: clean. Memory: open fridge; get food. Control Mode: ee."
    with pytest.raises(TypeError):
        PromptContext("clean", language_memory="open fridge")
    with pytest.raises(ValueError, match="Control mode"):
        PromptContext("clean", control_mode="cartesian")


def test_seeded_dropout_reproducible_and_control_always_retained():
    context = PromptContext("clean", "pick", EpisodeMetadata(2000, 5, mistake=False), "ee")
    first, second = np.random.default_rng(14), np.random.default_rng(14)
    for _ in range(50):
        sample = sample_context_dropout(context, first)
        assert sample == sample_context_dropout(context, second)
        assert sample.context.control_mode == "ee"
        assert sample.context.task == "clean"
        if not sample.keep_goals:
            assert sample.context.subtask == "pick"


def test_dropout_paper_rates():
    context = PromptContext("clean", "pick", EpisodeMetadata(2000, 5, mistake=False))
    rng = np.random.default_rng(831)
    samples = [sample_context_dropout(context, rng) for _ in range(20000)]
    goal_samples = [sample for sample in samples if sample.keep_goals]
    assert len(goal_samples) / len(samples) == pytest.approx(0.25, abs=0.015)
    assert np.mean([sample.context.subtask is None for sample in goal_samples]) == pytest.approx(0.3, abs=0.02)
    assert np.mean([sample.keep_history for sample in samples]) == pytest.approx(0.7, abs=0.015)
    assert np.mean([sample.keep_rear for sample in samples]) == pytest.approx(0.7, abs=0.015)
    for field in ("speed", "quality", "mistake"):
        assert np.mean([getattr(sample.context.metadata, field) is not None for sample in samples]) == pytest.approx(
            0.85 * 0.95, abs=0.015
        )


def test_dropout_all_metadata_and_subtask_only_with_goals():
    context = PromptContext("clean", "pick", EpisodeMetadata(2000, 5, mistake=False))
    config = DropoutConfig(goal_keep_probability=1, subtask_drop_with_goal=1, metadata_drop_probability=1)
    sample = sample_context_dropout(context, np.random.default_rng(1), goals_available=False, config=config)
    assert sample.context.metadata == EpisodeMetadata()
    assert sample.context.subtask == "pick"
    assert not sample.keep_goals
