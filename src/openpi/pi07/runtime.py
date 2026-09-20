"""Framework-independent, nonblocking implementation of pi0.7 Algorithm 1.

Call ``step`` once per control tick and dispatch its returned action externally.
This module never connects to or commands hardware. The language policy, world
model, and action policy are injected callables; it does not supply their trained
weights. A missing action (startup or an exhausted chunk) is reported as ``None``
instead of replaying an expired command.

Requests use the observation timestep as their action-chunk origin. Consequently,
if inference started at step 15 finishes at step 18, action 3 is used. The previous
chunk's remaining actions and an estimated delay are passed to the action callable
for real-time chunking (RTC). The callable must implement prefix conditioning;
merely using this scheduler does not add RTC to an arbitrary action model.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import CancelledError
from concurrent.futures import Executor
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
import copy
import dataclasses
import math
import time
from typing import Any


@dataclasses.dataclass(frozen=True)
class HistoryFrame:
    timestamp: float
    observation: Any


class ObservationHistory:
    """Bounded 1 Hz history, with the current observation always included.

    The six-frame budget includes the current frame. Older frames are sampled on
    a fixed cadence beginning with the first observation. Cadence samples less
    than one stride behind the current frame are omitted; no future frames or
    fabricated padding are returned. With irregular input times, gaps can exceed
    the stride. At most ``max_frames + 1`` observation objects are retained.
    """

    def __init__(self, max_frames: int = 6, stride_seconds: float = 1.0):
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if not math.isfinite(stride_seconds) or stride_seconds <= 0:
            raise ValueError("stride_seconds must be finite and positive")
        self.max_frames = max_frames
        self.stride_seconds = stride_seconds
        self._samples: deque[HistoryFrame] = deque(maxlen=max_frames)
        self._current: HistoryFrame | None = None

    def append(self, observation: Any, timestamp: float) -> None:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._current is not None and timestamp < self._current.timestamp:
            raise ValueError("observation timestamps must be nondecreasing")
        frame = HistoryFrame(timestamp, copy.deepcopy(observation))
        if not self._samples or timestamp - self._samples[-1].timestamp >= self.stride_seconds:
            self._samples.append(frame)
        self._current = frame

    def snapshot(self) -> tuple[HistoryFrame, ...]:
        if self._current is None:
            return ()
        earlier = [frame for frame in self._samples if self._current.timestamp - frame.timestamp >= self.stride_seconds]
        older_count = self.max_frames - 1
        return (*earlier[-older_count:], self._current) if older_count else (self._current,)

    def clear(self) -> None:
        self._samples.clear()
        self._current = None


@dataclasses.dataclass(frozen=True)
class RuntimeContext:
    task: str
    subtask: str | None
    goal_images: Any
    metadata: Any
    control_mode: str


@dataclasses.dataclass(frozen=True)
class GenerationRequest:
    observation: Any
    history: tuple[HistoryFrame, ...]
    context: RuntimeContext
    subtask_history: tuple[str, ...]
    timestep: int
    timestamp: float


@dataclasses.dataclass(frozen=True)
class PolicyRequest(GenerationRequest):
    previous_actions: tuple[Any, ...]
    inference_delay_steps: int


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    action_horizon: int = 50
    execute_steps: int = 25
    history_frames: int = 6
    history_stride_seconds: float = 1.0
    goal_refresh_seconds: float = 4.0
    # The paper leaves high-level query cadence unspecified; this is configurable.
    subtask_refresh_seconds: float = 1.0
    subtask_history_limit: int = 64
    max_inference_delay_steps: int = 12
    initial_inference_delay_steps: int = 0

    def __post_init__(self) -> None:
        if self.action_horizon != 50 or self.execute_steps not in (15, 25):
            raise ValueError("pi0.7 uses horizon 50 and execute_steps 15 or 25")
        for name in ("history_frames", "subtask_history_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("history_stride_seconds", "goal_refresh_seconds", "subtask_refresh_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_inference_delay_steps", "initial_inference_delay_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.initial_inference_delay_steps > self.max_inference_delay_steps:
            raise ValueError("initial inference delay cannot exceed the maximum")


class RuntimeGenerationError(RuntimeError):
    """An injected model failed; the original exception is attached as its cause."""


@dataclasses.dataclass
class _Job:
    stream: str
    serial: int
    episode: int
    intent: int
    language_generation: int
    request: GenerationRequest
    future: Future


def _default_intent_key(subtask: str | None) -> Hashable:
    return " ".join((subtask or "").split()).casefold()


class Pi07Runtime:
    """Asynchronous context generation and timestep-aligned action scheduling.

    All public methods must be called from the same control thread. Model calls
    run in workers and must treat requests as read-only. Observations are copied
    on ingestion so that camera buffers can safely be reused by the caller.

    ``intent_key`` can implement a caller's semantic intent identity. By default
    normalized subtask text determines identity; semantic paraphrase detection is
    not claimed. Coaching suppresses automatic language updates until
    ``resume_autonomy``. Intent changes invalidate old goals and action chunks.
    ``reset`` also prevents previous-episode work from becoming current.

    Timestamps must share the injected clock's monotonic timebase. No sleeping is
    performed here; the caller is responsible for its control rate. The default
    12-step RTC delay range matches the paper's training recipe at 50 Hz. If an
    observed delay exceeds that range, scheduling remains aligned but the model's
    delay-conditioning training range has been exceeded.
    """

    def __init__(
        self,
        action_policy: Callable[[PolicyRequest], Sequence[Any]],
        *,
        task: str,
        metadata: Any = None,
        control_mode: str = "joint",
        initial_subtask: str | None = None,
        high_level_policy: Callable[[GenerationRequest], str | None] | None = None,
        world_model: Callable[[GenerationRequest], Any] | None = None,
        config: RuntimeConfig | None = None,
        executor: Executor | None = None,
        clock: Callable[[], float] = time.monotonic,
        intent_key: Callable[[str | None], Hashable] = _default_intent_key,
    ):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty string")
        if control_mode not in ("joint", "ee"):
            raise ValueError("control_mode must be 'joint' or 'ee'")
        self.config = config or RuntimeConfig()
        self._action_policy = action_policy
        self._high_level_policy = high_level_policy
        self._world_model = world_model
        self._clock = clock
        self._intent_key = intent_key
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=3, thread_name_prefix="pi07")
        self._history = ObservationHistory(self.config.history_frames, self.config.history_stride_seconds)
        self._language_history: deque[str] = deque(maxlen=self.config.subtask_history_limit)
        self._context = RuntimeContext(task, None, None, copy.deepcopy(metadata), control_mode)
        self._jobs: list[_Job] = []
        self._current_jobs: dict[str, _Job] = {}
        self._accepted: dict[str, int] = {}
        self._serial = 0
        self._episode = 0
        self._intent = 0
        self._language_generation = 0
        self._manual_coaching = False
        self._closed = False
        self._last_timestep: int | None = None
        self._last_timestamp: float | None = None
        self._last_policy_start: int | None = None
        self._last_language_start: float | None = None
        self._last_goal_produced: float | None = None
        self._actions: tuple[Any, ...] = ()
        self._action_start = 0
        self._delay_steps = self.config.initial_inference_delay_steps
        self._apply_subtask(initial_subtask)

    @property
    def context(self) -> RuntimeContext:
        return self._context

    @property
    def subtask_history(self) -> tuple[str, ...]:
        """Issued semantic instructions, including the current one."""
        return tuple(self._language_history)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")

    def _invalidate(self, *streams: str) -> None:
        for stream in streams:
            job = self._current_jobs.pop(stream, None)
            if job is not None:
                job.future.cancel()

    def _apply_subtask(self, subtask: str | None) -> None:
        if subtask is not None and (not isinstance(subtask, str) or not subtask.strip()):
            raise ValueError("subtask must be a nonempty string or None")
        changed = self._intent_key(subtask) != self._intent_key(self._context.subtask)
        self._context = dataclasses.replace(self._context, subtask=subtask)
        if not changed:
            return
        self._intent += 1
        self._invalidate("goal", "policy", "language")
        self._context = dataclasses.replace(self._context, goal_images=None)
        self._last_goal_produced = None
        self._last_policy_start = None
        self._actions = ()
        if subtask is not None:
            self._language_history.append(subtask)

    def coach(self, subtask: str | None) -> None:
        """Supply a manual instruction and suspend high-level policy generation."""
        self._check_open()
        # Validate before changing the autonomous/manual state.
        self._apply_subtask(subtask)
        self._manual_coaching = True
        self._language_generation += 1
        self._invalidate("language")

    def resume_autonomy(self) -> None:
        self._check_open()
        self._manual_coaching = False
        self._language_generation += 1
        self._last_language_start = None
        self._invalidate("language")

    def reset(self, *, task: str | None = None, initial_subtask: str | None = None) -> None:
        """Start an episode, retaining the configured metadata and control mode."""
        self._check_open()
        if task is not None and (not isinstance(task, str) or not task.strip()):
            raise ValueError("task must be a nonempty string")
        if initial_subtask is not None and (not isinstance(initial_subtask, str) or not initial_subtask.strip()):
            raise ValueError("initial_subtask must be a nonempty string or None")
        self._episode += 1
        self._language_generation += 1
        self._invalidate("language", "goal", "policy")
        self._history.clear()
        self._language_history.clear()
        self._accepted.clear()
        self._context = dataclasses.replace(
            self._context, task=task or self._context.task, subtask=None, goal_images=None
        )
        self._manual_coaching = False
        self._last_timestep = None
        self._last_timestamp = None
        self._last_policy_start = None
        self._last_language_start = None
        self._last_goal_produced = None
        self._actions = ()
        self._delay_steps = self.config.initial_inference_delay_steps
        self._apply_subtask(initial_subtask)

    def _submit(self, stream: str, model: Callable, request: GenerationRequest) -> None:
        def invoke() -> tuple[Any, float]:
            return model(request), self._clock()

        self._serial += 1
        future = self._executor.submit(invoke)
        job = _Job(stream, self._serial, self._episode, self._intent, self._language_generation, request, future)
        self._jobs.append(job)
        self._current_jobs[stream] = job

    def _is_current(self, job: _Job) -> bool:
        return (
            job.episode == self._episode
            and job.intent == self._intent
            and job.serial > self._accepted.get(job.stream, -1)
            and (
                job.stream != "language"
                or (not self._manual_coaching and job.language_generation == self._language_generation)
            )
        )

    def _collect(self, timestep: int, *, apply_results: bool = True) -> None:
        pending: list[_Job] = []
        failures: list[tuple[_Job, Exception]] = []
        # Process language first so that simultaneously completed old-intent
        # goals/actions cannot be used for a newly accepted instruction.
        for job in sorted(self._jobs, key=lambda item: (item.stream != "language", item.serial)):
            if not job.future.done():
                pending.append(job)
                continue
            if self._current_jobs.get(job.stream) is job:
                self._current_jobs.pop(job.stream)
            try:
                result, produced_at = job.future.result()
                if not apply_results or not self._is_current(job):
                    continue
                if job.stream == "language":
                    self._apply_subtask(result)
                elif job.stream == "goal":
                    self._context = dataclasses.replace(self._context, goal_images=copy.deepcopy(result))
                    self._last_goal_produced = produced_at
                else:
                    actions = tuple(result)
                    if len(actions) != self.config.action_horizon:
                        raise ValueError(f"action policy must return {self.config.action_horizon} actions")
                    latency = timestep - job.request.timestep
                    self._delay_steps = min(latency, self.config.max_inference_delay_steps)
                    if latency < len(actions):
                        self._actions = copy.deepcopy(actions)
                        self._action_start = job.request.timestep
                self._accepted[job.stream] = job.serial
            except CancelledError:
                continue
            except Exception as exc:
                failures.append((job, exc))
        self._jobs = pending
        if failures:
            job, exc = failures[0]
            # Preserve additional failures instead of silently dropping them.
            error = RuntimeGenerationError(f"{job.stream} generation {job.serial} failed: {exc}")
            for other_job, other_exc in failures[1:]:
                error.add_note(f"{other_job.stream} generation {other_job.serial} also failed: {other_exc}")
            raise error from exc

    def _remaining(self, timestep: int) -> tuple[Any, ...]:
        index = timestep - self._action_start
        if not self._actions or index < 0 or index >= len(self._actions):
            return ()
        return self._actions[index:]

    def step(self, observation: Any, *, timestep: int, timestamp: float | None = None) -> Any | None:
        """Poll completed models, queue due work, and return this tick's action.

        ``timestep`` must strictly increase within each episode. No model call
        blocks this method; even instantaneous results are observed next tick.
        """
        self._check_open()
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError("timestep must be a nonnegative integer")
        if self._last_timestep is not None and timestep <= self._last_timestep:
            raise ValueError("timestep must strictly increase; call reset for a new episode")
        now = self._clock() if timestamp is None else timestamp
        if not math.isfinite(now) or (self._last_timestamp is not None and now < self._last_timestamp):
            raise ValueError("timestamp must be finite and nondecreasing")
        self._history.append(observation, now)
        self._last_timestep = timestep
        self._last_timestamp = now
        self._collect(timestep)
        history = self._history.snapshot()
        request = GenerationRequest(
            history[-1].observation, history, self._context, self.subtask_history, timestep, now
        )
        if (
            self._high_level_policy is not None
            and not self._manual_coaching
            and "language" not in self._current_jobs
            and (
                self._last_language_start is None
                or now - self._last_language_start >= self.config.subtask_refresh_seconds
            )
        ):
            self._submit("language", self._high_level_policy, request)
            self._last_language_start = now
        if (
            self._world_model is not None
            and self._context.subtask is not None
            and "goal" not in self._current_jobs
            and (self._last_goal_produced is None or now - self._last_goal_produced >= self.config.goal_refresh_seconds)
        ):
            self._submit("goal", self._world_model, request)
        remaining = self._remaining(timestep)
        if "policy" not in self._current_jobs and (
            not remaining
            or self._last_policy_start is None
            or timestep - self._last_policy_start >= self.config.execute_steps
        ):
            policy_request = PolicyRequest(
                observation=request.observation,
                history=history,
                context=self._context,
                subtask_history=self.subtask_history,
                timestep=timestep,
                timestamp=now,
                previous_actions=remaining,
                inference_delay_steps=min(self._delay_steps, len(remaining)),
            )
            self._submit("policy", self._action_policy, policy_request)
            self._last_policy_start = timestep
        return remaining[0] if remaining else None

    def close(self, *, wait: bool = True) -> None:
        """Cancel queued work and optionally join running calls, surfacing errors.

        Python threads cannot forcibly interrupt a running model. ``wait=False``
        returns promptly, but the injected model must still finish its own work.
        An externally supplied executor remains owned by its caller.
        """
        if self._closed:
            return
        self._closed = True
        for job in self._jobs:
            job.future.cancel()
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)
        elif wait:
            wait_futures([job.future for job in self._jobs])
        self._collect(self._last_timestep or 0, apply_results=False)

    def __enter__(self) -> Pi07Runtime:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except RuntimeGenerationError:
            if exc_value is None:
                raise
