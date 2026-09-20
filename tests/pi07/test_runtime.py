"""Synthetic scheduling checks; no robot, network, model weights, or sleeps."""

from concurrent.futures import Executor
from concurrent.futures import Future
import dataclasses
import threading

import pytest

from openpi.pi07.runtime import ObservationHistory
from openpi.pi07.runtime import Pi07Runtime
from openpi.pi07.runtime import RuntimeConfig
from openpi.pi07.runtime import RuntimeGenerationError


class ManualExecutor(Executor):
    def __init__(self):
        self.calls = []

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        # Mark running so stale inferences cannot be canceled, as on a real GPU.
        future.set_running_or_notify_cancel()
        self.calls.append((future, fn, args, kwargs))
        return future

    def complete(self, index):
        future, fn, args, kwargs = self.calls[index]
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)

    def finish(self):
        for index, (future, *_rest) in enumerate(self.calls):
            if not future.done():
                self.complete(index)


@dataclasses.dataclass
class Clock:
    now: float = 0.0

    def __call__(self):
        return self.now


def _actions(request):
    return [(request.timestep, index) for index in range(50)]


def test_history_current_included_bounded_and_copied():
    history = ObservationHistory()
    value = {"state": [0]}
    history.append(value, 0.0)
    value["state"][0] = 99
    assert history.snapshot()[0].observation == {"state": [0]}
    for index in range(1, 2001):
        history.append(index, index / 50)
    assert [frame.timestamp for frame in history.snapshot()] == [35, 36, 37, 38, 39, 40]
    history.append("current", 40.2)
    assert [frame.timestamp for frame in history.snapshot()] == [35, 36, 37, 38, 39, 40.2]
    history.clear()
    assert history.snapshot() == ()


def test_history_one_frame_and_bad_time():
    history = ObservationHistory(max_frames=1)
    history.append("old", 1)
    history.append("new", 2)
    assert [frame.observation for frame in history.snapshot()] == ["new"]
    with pytest.raises(ValueError, match="nondecreasing"):
        history.append("past", 0)
    with pytest.raises(ValueError, match="finite"):
        history.append("bad", float("nan"))


@pytest.mark.parametrize("execute_steps", [15, 25])
def test_delay_alignment_and_rtc_remaining_prefix(execute_steps):
    executor = ManualExecutor()
    requests = []

    def policy(request):
        requests.append(request)
        return _actions(request)

    runtime = Pi07Runtime(policy, task="fold", executor=executor, config=RuntimeConfig(execute_steps=execute_steps))
    assert runtime.step({}, timestep=0, timestamp=0) is None
    executor.complete(0)
    assert runtime.step({}, timestep=3, timestamp=0.06) == (0, 3)
    assert runtime.step({}, timestep=execute_steps, timestamp=1) == (0, execute_steps)
    executor.complete(1)
    assert requests[1].previous_actions == tuple((0, index) for index in range(execute_steps, 50))
    assert requests[1].inference_delay_steps == 3
    assert runtime.step({}, timestep=execute_steps + 2, timestamp=2) == (execute_steps, 2)
    runtime.close()


def test_expired_chunk_is_not_replayed_and_late_result_is_not_used():
    executor = ManualExecutor()
    runtime = Pi07Runtime(_actions, task="fold", executor=executor)
    runtime.step({}, timestep=0, timestamp=0)
    executor.complete(0)
    assert runtime.step({}, timestep=51, timestamp=1.02) is None
    assert len(executor.calls) == 2
    executor.complete(1)
    assert runtime.step({}, timestep=52, timestamp=1.04) == (51, 1)
    runtime.close()


def test_goal_refresh_timer_starts_when_generated_not_requested():
    executor = ManualExecutor()
    clock = Clock()
    goal_requests = []

    def world_model(request):
        goal_requests.append(request)
        return {"front": request.timestamp}

    runtime = Pi07Runtime(
        _actions,
        task="fold",
        initial_subtask="grasp corner",
        world_model=world_model,
        executor=executor,
        clock=clock,
    )
    runtime.step({}, timestep=0)
    clock.now = 2
    executor.complete(0)  # Goal was requested at 0, produced at 2.
    executor.complete(1)
    runtime.step({}, timestep=1)
    assert runtime.context.goal_images == {"front": 0}
    clock.now = 5.9
    runtime.step({}, timestep=2)
    assert len(executor.calls) == 2
    clock.now = 6
    runtime.step({}, timestep=3)
    assert len(executor.calls) == 3
    assert len(goal_requests) == 1  # New request is asynchronous.
    executor.finish()
    runtime.close()


def test_old_goal_and_action_cannot_replace_new_intent():
    executor = ManualExecutor()
    runtime = Pi07Runtime(
        lambda request: [(request.context.subtask, index) for index in range(50)],
        task="fold",
        initial_subtask="grasp",
        world_model=lambda request: request.context.subtask,
        executor=executor,
        clock=lambda: 0,
    )
    runtime.step({}, timestep=0, timestamp=0)  # Old goal 0, action 1 remain running.
    runtime.coach("lift")
    runtime.step({}, timestep=1, timestamp=0.02)  # New goal 2, action 3.
    executor.complete(2)
    executor.complete(3)
    assert runtime.step({}, timestep=2, timestamp=0.04) == ("lift", 1)
    assert runtime.context.goal_images == "lift"
    executor.complete(0)
    executor.complete(1)
    assert runtime.step({}, timestep=3, timestamp=0.06) == ("lift", 2)
    assert runtime.context.goal_images == "lift"
    assert runtime.subtask_history == ("grasp", "lift")
    runtime.close()


def test_old_episode_result_is_rejected_even_with_same_task_and_subtask():
    executor = ManualExecutor()
    runtime = Pi07Runtime(_actions, task="fold", initial_subtask="grasp", executor=executor)
    runtime.step({}, timestep=0, timestamp=0)
    runtime.reset(initial_subtask="grasp")
    runtime.step({}, timestep=0, timestamp=0)
    executor.complete(0)
    assert runtime.step({}, timestep=1, timestamp=1) is None
    executor.complete(1)
    assert runtime.step({}, timestep=2, timestamp=2) == (0, 2)
    assert runtime.subtask_history == ("grasp",)
    runtime.close()


def test_coaching_cannot_be_overwritten_by_running_language_request():
    executor = ManualExecutor()
    runtime = Pi07Runtime(
        _actions,
        task="fold",
        high_level_policy=lambda request: "automated instruction",
        executor=executor,
        clock=lambda: 0,
    )
    runtime.step({}, timestep=0, timestamp=0)
    runtime.coach("human instruction")
    executor.complete(0)
    executor.complete(1)
    assert runtime.step({}, timestep=1, timestamp=1) is None
    assert runtime.context.subtask == "human instruction"
    executor.complete(2)
    runtime.resume_autonomy()
    runtime.step({}, timestep=2, timestamp=2)
    executor.complete(3)
    runtime.step({}, timestep=3, timestamp=3)
    assert runtime.context.subtask == "automated instruction"
    executor.finish()
    runtime.close()


def test_matching_goals_remain_available_while_refresh_runs():
    executor = ManualExecutor()
    clock = Clock()
    runtime = Pi07Runtime(
        _actions,
        task="fold",
        initial_subtask="grasp",
        world_model=lambda request: request.timestamp,
        executor=executor,
        clock=clock,
    )
    runtime.step({}, timestep=0)
    executor.complete(0)
    executor.complete(1)
    clock.now = 4
    runtime.step({}, timestep=1)
    assert runtime.context.goal_images == 0
    assert len(executor.calls) == 3
    clock.now = 4.1
    runtime.step({}, timestep=2)
    assert runtime.context.goal_images == 0
    executor.complete(2)
    runtime.step({}, timestep=3)
    assert runtime.context.goal_images == 4
    runtime.close()


def test_equivalent_intent_does_not_regenerate_goals():
    executor = ManualExecutor()
    runtime = Pi07Runtime(
        _actions,
        task="fold",
        initial_subtask="grasp corner",
        world_model=lambda request: "goal",
        executor=executor,
    )
    runtime.step({}, timestep=0, timestamp=0)
    executor.finish()
    runtime.step({}, timestep=1, timestamp=1)
    runtime.coach(" Grasp   corner ")
    runtime.step({}, timestep=2, timestamp=2)
    assert runtime.context.goal_images == "goal"
    assert len(executor.calls) == 2
    runtime.close()


def test_generation_error_surfaces_with_original_cause():
    executor = ManualExecutor()

    def broken_policy(request):
        raise LookupError("missing checkpoint")

    runtime = Pi07Runtime(broken_policy, task="fold", executor=executor)
    runtime.step({}, timestep=0, timestamp=0)
    executor.complete(0)
    with pytest.raises(RuntimeGenerationError, match=r"policy.*missing checkpoint") as failure:
        runtime.step({}, timestep=1, timestamp=1)
    assert isinstance(failure.value.__cause__, LookupError)
    runtime.close()


def test_invalid_chunk_length_surfaces():
    executor = ManualExecutor()
    runtime = Pi07Runtime(lambda request: [0] * 49, task="fold", executor=executor)
    runtime.step({}, timestep=0, timestamp=0)
    executor.complete(0)
    with pytest.raises(RuntimeGenerationError, match="50 actions"):
        runtime.step({}, timestep=1, timestamp=1)
    runtime.close()


def test_close_surfaces_completed_error_and_prevents_further_steps():
    executor = ManualExecutor()

    def fail(request):
        raise ValueError("generation failed")

    runtime = Pi07Runtime(fail, task="fold", executor=executor)
    runtime.step({}, timestep=0, timestamp=0)
    executor.complete(0)
    with pytest.raises(RuntimeGenerationError, match="generation failed"):
        runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.step({}, timestep=1, timestamp=1)
    runtime.close()


def test_real_workers_are_nonblocking_and_close_joins():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def policy(request):
        started.set()
        assert release.wait(timeout=5)
        finished.set()
        return _actions(request)

    runtime = Pi07Runtime(policy, task="fold")
    try:
        assert runtime.step({}, timestep=0) is None
        assert started.wait(timeout=5)
        # A second tick returns while the worker is still waiting.
        assert runtime.step({}, timestep=1) is None
        assert not finished.is_set()
    finally:
        release.set()
        runtime.close()
    assert finished.is_set()


@pytest.mark.parametrize(
    "options",
    [
        {"action_horizon": 49},
        {"execute_steps": 10},
        {"goal_refresh_seconds": 0},
        {"history_frames": 0},
        {"subtask_history_limit": 0},
        {"initial_inference_delay_steps": 13},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError, match=r"pi0.7 uses|must be|cannot exceed"):
        RuntimeConfig(**options)
