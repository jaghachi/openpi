import numpy as np
import pytest

from openpi.pi07.context import DropoutConfig
from openpi.pi07.data import Episode
from openpi.pi07.data import EpisodeDataset
from openpi.pi07.data import SamplingConfig
from openpi.pi07.data import collate_examples
from openpi.pi07.data import sample_example
from openpi.pi07.data import sample_goal_index
from openpi.pi07.data import sample_history_indices
from openpi.pi07.data import subtask_memory


def episode(length=12, views=4, **kwargs):
    images = np.broadcast_to(
        np.arange(length, dtype=np.uint8)[:, None, None, None, None], (length, views, 4, 6, 3)
    ).copy()
    return Episode(
        images,
        np.arange(length * 3).reshape(length, 3),
        np.arange(length * 2).reshape(length, 2),
        "fold the shirt",
        "joint",
        2,
        **kwargs,
    )


def test_history_is_one_second_spaced_and_stays_in_episode():
    item = episode()
    config = SamplingConfig(image_size=4)
    indices, mask = sample_history_indices(item, 5, config)
    assert indices.tolist() == [0, 0, 0, 1, 3, 5]
    assert mask.tolist() == [False, False, False, True, True, True]
    dataset = EpisodeDataset([episode(), episode()], config=config, training=False)
    sample = dataset[12]
    assert sample["history_indices"].tolist() == [0] * 6
    assert sample["state_mask"].tolist() == [False] * 5 + [True]
    np.testing.assert_array_equal(sample["states"][:-1], 0)


def test_irregular_history_never_selects_a_frame_after_target():
    item = episode(length=5, timestamps=np.array([0, 0.4, 1.3, 2.1, 3.5]))
    indices, mask = sample_history_indices(item, 4, SamplingConfig(history_frames=4))
    assert indices.tolist() == [1, 2, 3, 4]
    assert mask.all()


def test_goal_sampling_respects_episode_and_four_second_horizon():
    ends = np.array([5] * 6 + [11] * 6)
    item = episode(segment_end=ends)
    rng = np.random.default_rng(9)
    uniform = SamplingConfig(segment_end_probability=0, future_goal_seconds=1)
    seen = {sample_goal_index(item, 4, rng, uniform) for _ in range(200)}
    # Uniform goals can cross the segment ending at frame 5; only the 25%
    # segment-end branch is tied to the semantic boundary.
    assert seen == {4, 5, 6}
    assert sample_goal_index(item, 4, rng, SamplingConfig(segment_end_probability=1)) == 5
    assert sample_goal_index(item, 11, rng, uniform) == 11
    long_item = episode(length=30)
    seen = {sample_goal_index(long_item, 2, rng, SamplingConfig(segment_end_probability=0)) for _ in range(500)}
    assert seen == set(range(2, 11))


def test_goal_sampling_mixture_matches_paper():
    item = episode(length=40)
    config = SamplingConfig()
    rng = np.random.default_rng(521)
    indices = [sample_goal_index(item, 0, rng, config) for _ in range(10000)]
    assert np.mean(np.array(indices) == 39) == pytest.approx(0.25, abs=0.015)


def test_action_chunks_pad_only_after_episode_with_coordinate_masks():
    mask = np.ones((12, 2), dtype=bool)
    mask[:, 1] = False
    sample = sample_example(
        episode(action_mask=mask), 10, np.random.default_rng(0), config=SamplingConfig(image_size=4), training=False
    )
    assert sample["actions"].shape == (50, 2)
    assert sample["action_mask"].sum() == 2
    np.testing.assert_array_equal(sample["actions"][:2, 0], [20, 22])
    np.testing.assert_array_equal(sample["actions"][2:], 0)
    np.testing.assert_array_equal(sample["actions"][:, 1], 0)


def test_dropout_masks_history_states_and_rear_but_retains_current():
    config = SamplingConfig(
        image_size=4,
        dropout=DropoutConfig(goal_keep_probability=0, history_drop_probability=1, rear_drop_probability=1),
    )
    sample = sample_example(episode(), 11, np.random.default_rng(0), config=config)
    assert sample["state_mask"].tolist() == [False] * 5 + [True]
    assert sample["image_mask"].sum() == 3
    assert not sample["image_mask"][3].any()
    assert not sample["goal_mask"].any()
    np.testing.assert_array_equal(sample["images"][~sample["image_mask"]], 0)
    np.testing.assert_array_equal(sample["goal_images"], 0)


def test_resize_normalization_and_goal_rear_exclusion():
    sample = sample_example(
        episode(), 11, np.random.default_rng(0), config=SamplingConfig(image_size=8), training=False
    )
    assert sample["images"].shape == (4, 6, 3, 8, 8)
    assert sample["goal_images"].shape == (3, 3, 8, 8)
    np.testing.assert_allclose(sample["images"][0, -1], 11 / 255, atol=1e-7)
    assert sample["images"].dtype == np.float32


def test_generated_goal_selection_is_explicit():
    goals = np.full((12, 3, 4, 6, 3), 255, dtype=np.uint8)
    sample = sample_example(
        episode(generated_goals=goals),
        0,
        np.random.default_rng(0),
        config=SamplingConfig(image_size=4, generated_goal_probability=1),
        training=False,
    )
    assert sample["generated_goal"]
    np.testing.assert_array_equal(sample["goal_images"], 1)


def test_npz_roundtrip_and_scalar_annotations(tmp_path):
    item = episode(subtask="fold left sleeve", quality=np.int64(5), mistake=False)
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        images=item.images,
        states=item.states,
        actions=item.actions,
        task=item.task,
        control_mode="joint",
        fps=2.0,
        subtask="fold left sleeve",
        quality=5,
        mistake=False,
    )
    dataset = EpisodeDataset.from_npz([path], config=SamplingConfig(image_size=4), training=False)
    assert len(dataset) == 12
    assert "Subtask: fold left sleeve." in dataset[0]["prompt"]
    assert "Mistake: false." in dataset[-1]["prompt"]
    with pytest.raises(IndexError):
        dataset[12]


def test_subtask_memory_keeps_nonconsecutive_repetitions_without_label_leakage():
    labels = np.array(
        ["open", "open", "grasp", "grasp", "open", "open", "place", "place", "close", "close", "close", "close"]
    )
    item = episode(subtask=labels, segment_end=np.array([1, 1, 3, 3, 5, 5, 7, 7, 11, 11, 11, 11]))
    assert subtask_memory(item, 0) == ()
    assert subtask_memory(item, 1) == ()
    assert subtask_memory(item, 2) == ("open",)
    assert subtask_memory(item, 5) == ("open", "grasp")
    assert subtask_memory(item, 6) == ("open", "grasp", "open")
    assert subtask_memory(item, 11, limit=2) == ("open", "place")
    assert subtask_memory(item, 11, limit=0) == ()
    sample = sample_example(item, 6, np.random.default_rng(0), config=SamplingConfig(image_size=4), training=False)
    assert sample["context"].language_memory == ("open", "grasp", "open")
    assert "Memory: open; grasp; open." in sample["prompt"]
    assert "close" not in sample["prompt"]


def test_subtask_memory_deduplicates_consecutive_segments_and_is_bounded():
    labels = np.repeat(np.array([f"instruction {index}" for index in range(80)]), 2)
    item = episode(length=len(labels), subtask=labels)
    sample = sample_example(item, 159, np.random.default_rng(0), config=SamplingConfig(image_size=4), training=False)
    assert len(sample["context"].language_memory) == 64
    assert sample["context"].language_memory[0] == "instruction 15"
    assert sample["context"].language_memory[-1] == "instruction 78"
    assert subtask_memory(episode(subtask="same"), 11) == ()
    assert subtask_memory(episode(), 11) == ()


def test_subtask_memory_honors_configured_limit():
    item = episode(subtask=np.array(["a", "b", "c", "d"] * 3))
    config = SamplingConfig(image_size=4, language_memory_limit=1)
    sample = sample_example(item, 11, np.random.default_rng(0), config=config, training=False)
    assert sample["context"].language_memory == ("c",)


def test_dataset_seed_is_independent_of_access_order():
    dataset = EpisodeDataset([episode()], config=SamplingConfig(image_size=4), seed=8)
    first = dataset[9]
    dataset[2]
    repeated = dataset[9]
    assert first["prompt"] == repeated["prompt"]
    assert first["goal_index"] == repeated["goal_index"]
    np.testing.assert_array_equal(first["image_mask"], repeated["image_mask"])
    dataset.set_epoch(1)
    changed = dataset[9]
    assert (
        first["prompt"] != changed["prompt"]
        or first["goal_index"] != changed["goal_index"]
        or not np.array_equal(first["image_mask"], changed["image_mask"])
    )


def test_collate_pads_views_dimensions_and_tokens_without_losing_masks():
    config = SamplingConfig(image_size=4)
    samples = [
        sample_example(episode(views=views), 11, np.random.default_rng(0), config=config, training=False)
        for views in (1, 4)
    ]
    samples[0]["prompt"] = "a"
    encoded_actions = []

    def action_tokenizer(actions, mask):
        encoded_actions.append((actions, mask))
        return [6, 7, 8]

    batch = collate_examples(samples, lambda text: list(text.encode()), action_tokenizer=action_tokenizer)
    assert batch["images"].shape == (2, 4, 6, 3, 4, 4)
    assert not batch["image_mask"][0, 1:].any()
    assert batch["goal_mask"].sum(axis=1).tolist() == [1, 3]
    assert batch["text_mask"][0].sum() == 1
    assert batch["fast_token_ids"].tolist() == [[6, 7, 8]] * 2
    assert batch["fast_mask"].all()
    assert len(encoded_actions) == 2
    assert encoded_actions[0][1].sum() == 2
    assert set(batch) == {
        "images",
        "image_mask",
        "states",
        "state_mask",
        "goal_images",
        "goal_mask",
        "input_ids",
        "text_mask",
        "actions",
        "action_mask",
        "fast_token_ids",
        "fast_mask",
    }


def test_collate_missing_fast_is_empty_and_text_overflow_is_explicit():
    sample = sample_example(episode(), 0, np.random.default_rng(0), config=SamplingConfig(image_size=4))
    result = collate_examples([sample], lambda _: [1, 2])
    assert result["fast_token_ids"].shape == (1, 0)
    with pytest.raises(ValueError, match="truncate"):
        collate_examples([sample], lambda _: [1, 2], max_text_length=1)
    with pytest.raises(ValueError, match="FAST"):
        collate_examples([sample], lambda _: [1], action_tokenizer=lambda actions, mask: [1.9])


def test_collate_pads_to_model_dimensions_after_fast_encoding():
    sample = sample_example(episode(), 11, np.random.default_rng(0), config=SamplingConfig(image_size=4))
    encoded_shapes = []

    def encode(actions, mask):
        encoded_shapes.append((actions.shape, mask.shape))
        return [1, 9]

    batch = collate_examples(
        [sample],
        lambda _: [1],
        action_tokenizer=encode,
        target_state_dim=32,
        target_action_dim=32,
    )
    assert batch["states"].shape == (1, 6, 32)
    assert batch["actions"].shape == (1, 50, 32)
    assert batch["action_mask"].shape == (1, 50, 32)
    assert batch["action_mask"].sum() == 2
    np.testing.assert_array_equal(batch["states"][..., 3:], 0)
    np.testing.assert_array_equal(batch["actions"][..., 2:], 0)
    assert encoded_shapes == [((50, 2), (50, 2))]
    with pytest.raises(ValueError, match="exceeds"):
        collate_examples([sample], lambda _: [1], target_action_dim=1)


def test_collate_right_aligns_variable_history_at_current_frame():
    samples = [
        sample_example(
            episode(),
            11,
            np.random.default_rng(0),
            config=SamplingConfig(image_size=4, history_frames=count),
            training=False,
        )
        for count in (2, 6)
    ]
    batch = collate_examples(samples, lambda _: [1])
    assert batch["state_mask"][0].tolist() == [False, False, False, False, True, True]
    assert not batch["image_mask"][0, :, :4].any()
    np.testing.assert_array_equal(batch["states"][0, -1], batch["states"][1, -1])
    np.testing.assert_array_equal(batch["images"][0, :, -1], batch["images"][1, :, -1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mistake": "false"},
        {"quality": 4.8},
        {"fps": 0},
        {"segment_end": np.array([12] * 12)},
        {"segment_end": np.array([2] * 3 + [8] * 2 + [11] * 7)},
        {"timestamps": np.zeros(12)},
        {"action_mask": np.ones((12, 2))},
    ],
)
def test_episode_rejects_ambiguous_or_misaligned_annotations(kwargs):
    if "fps" in kwargs:
        item = episode()
        with pytest.raises((ValueError, TypeError)):
            Episode(item.images, item.states, item.actions, item.task, "joint", **kwargs)
    else:
        with pytest.raises((ValueError, TypeError)):
            episode(**kwargs)
