"""Tinker cookbook RL recipe. Import this module only with train extras installed."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits, TerminationRewardPolicy
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env, simple_tool_result, tool

from .episode import ELTEpisode
from .spec import TaskSpec


SYSTEM_PROMPT = """You are building one complete ELT pipeline. Inspect the task,
configure and run extraction/loading, write every requested transformation,
use execution feedback to repair failures, then call submit_pipeline. The
warehouse state, not your final explanation, determines reward. Ground truth
is private. Never claim success without running the pipeline."""


def _rollout_config(max_turns: int) -> RolloutConfig:
    return RolloutConfig(
        limits=RolloutLimits(max_turns=max_turns, max_trajectory_tokens=65536,
                             max_tool_calls=50, rollout_timeout_seconds=1800),
        parse_errors=ParseErrorPolicy(max_consecutive=2),
        termination=TerminationRewardPolicy(
            zero_reward_on_limit=True, skip_grading_on_timeout=True,
            grader_timeout_seconds=900,
        ),
        tool_execution="sequential",
    )


def _make_env(episode: ELTEpisode, model_name: str, max_turns: int) -> Env:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer_utils.get_tokenizer(model_name))

    @tool
    async def inspect_task():
        """Read the ELT task, source names, and required final models."""
        result = await asyncio.to_thread(episode.act, "inspect_task")
        return simple_tool_result(result.observation)

    @tool
    async def read_file(path: str):
        """Read a task file such as config.yaml, data_model.yaml, schemas/*.csv, or documentation/*.md."""
        result = await asyncio.to_thread(episode.act, "read_file", path=path)
        return simple_tool_result(result.observation)

    @tool
    async def write_model(model: str, sql: str):
        """Write one requested target model as a SELECT query."""
        result = await asyncio.to_thread(episode.act, "write_model", model=model, sql=sql)
        return simple_tool_result(result.observation)

    @tool
    async def run_transforms():
        """Execute the staged transformation models and return dbt/SQL feedback."""
        result = await asyncio.to_thread(episode.act, "run_transforms")
        return simple_tool_result(result.observation)

    @tool
    async def preview_sql(sql: str):
        """Run a read-only SELECT to inspect up to 20 warehouse rows."""
        result = await asyncio.to_thread(episode.act, "preview_sql", sql=sql)
        return simple_tool_result(result.observation)

    @tool
    async def submit_pipeline():
        """Finish the episode and privately grade the resulting warehouse state."""
        result = await asyncio.to_thread(episode.act, "submit_pipeline")
        return simple_tool_result(result.observation, should_stop=True)

    tools = [inspect_task, read_file, write_model, run_transforms, submit_pipeline]
    if episode.spec.destination == "duckdb":
        @tool
        async def configure_el(tables: list[str]):
            """Choose task-declared CSV source tables and load them into DuckDB."""
            result = await asyncio.to_thread(episode.act, "configure_el", tables=tables)
            return simple_tool_result(result.observation)
        tools[2:2] = [configure_el, preview_sql]
    else:
        @tool
        async def write_terraform(filename: str, content: str):
            """Configure Airbyte source, destination, and connections in a new elt/*.tf file; use config.yaml values through Terraform yamldecode."""
            result = await asyncio.to_thread(episode.act, "configure_el", filename=filename, content=content)
            return simple_tool_result(result.observation)

        @tool
        async def run_el():
            """Run terraform init/apply, trigger Airbyte syncs, and wait for jobs."""
            result = await asyncio.to_thread(episode.act, "run_el")
            return simple_tool_result(result.observation)

        @tool
        async def preview_table(table: str):
            """Inspect up to 20 rows of a declared raw or target table in this task."""
            result = await asyncio.to_thread(episode.act, "preview_table", table=table)
            return simple_tool_result(result.observation)
        tools[2:2] = [write_terraform, run_el, preview_table]

    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[item.to_spec() for item in tools], system_prompt=SYSTEM_PROMPT,
    )
    messages = prefix + [{"role": "user", "content": episode.spec.public_summary()}]

    async def reward_fn(_history):
        # The history is not trusted as grading evidence; only the executed state is.
        reward, metrics = episode.last_score, episode.last_metrics or {"submitted": 0.0}
        if episode.spec.destination == "duckdb":
            episode.close()
        return reward, metrics

    return build_agent_tool_env(
        renderer=renderer, tools=tools, initial_messages=messages,
        reward_fn=reward_fn, rollout_config=_rollout_config(max_turns), max_turns=max_turns,
        max_trajectory_tokens=65536, max_generation_tokens=4096,
    )


_OFFICIAL_LOCK = asyncio.Lock()


class _SerializedOfficialEnv(Env):
    """Serialize same-task official rollouts so reset/sync/grading cannot collide."""

    def __init__(self, spec: TaskSpec, run_root: Path, model_name: str, max_turns: int):
        self.spec, self.run_root = spec, run_root
        self.model_name, self.max_turns = model_name, max_turns
        self.episode: ELTEpisode | None = None
        self.inner: Env | None = None
        self._held = False
        self.rollout_limits = _rollout_config(max_turns).limits
        self._max_tool_calls: int | None = None

    async def initial_observation(self):
        await _OFFICIAL_LOCK.acquire()
        self._held = True
        try:
            self.episode = await asyncio.to_thread(ELTEpisode, self.spec, self.run_root)
            self.inner = _make_env(self.episode, self.model_name, self.max_turns)
            if self._max_tool_calls is not None:
                self.inner.set_max_tool_calls(self._max_tool_calls)
            return await self.inner.initial_observation()
        except BaseException:
            self._finish()
            raise

    def set_max_tool_calls(self, n: int) -> None:
        self._max_tool_calls = n
        if self.inner is not None:
            self.inner.set_max_tool_calls(n)

    async def step(self, action, *, extra=None):
        assert self.inner is not None
        try:
            result = await self.inner.step(action, extra=extra)
            if result.episode_done:
                self._finish()
            return result
        except BaseException:
            self._finish()
            raise

    def _finish(self):
        try:
            if self.episode is not None:
                self.episode.close()
                self.episode = None
        finally:
            if self._held:
                _OFFICIAL_LOCK.release()
                self._held = False


class ELTGroupBuilder(EnvGroupBuilder):
    def __init__(self, spec: TaskSpec, run_root: Path, model_name: str, group_size: int, max_turns: int):
        self.spec, self.run_root = spec, run_root
        self.model_name, self.group_size, self.max_turns = model_name, group_size, max_turns

    async def make_envs(self) -> Sequence[Env]:
        if self.spec.destination == "snowflake":
            return [_SerializedOfficialEnv(self.spec, self.run_root, self.model_name, self.max_turns)
                    for _ in range(self.group_size)]
        episodes = [ELTEpisode(self.spec, self.run_root) for _ in range(self.group_size)]
        return [_make_env(ep, self.model_name, self.max_turns) for ep in episodes]

    def logging_tags(self) -> list[str]:
        return [self.spec.task_id, self.spec.destination]


class ELTDataset(RLDataset):
    def __init__(self, groups: list[ELTGroupBuilder]):
        self.groups = groups

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [self.groups[index]]

    def __len__(self) -> int:
        return len(self.groups)


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    run_root: str
    fixture_dir: str | None = None
    official_repo: str | None = None
    task_id: str = "books"
    inputs_dir: str | None = None
    ground_truth_dir: str | None = None
    credential_path: str | None = None
    group_size: int = 4
    max_turns: int = 12

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        if self.official_repo:
            spec = TaskSpec.official_snowflake(
                Path(self.official_repo), self.task_id,
                inputs_dir=Path(self.inputs_dir) if self.inputs_dir else None,
                ground_truth_dir=Path(self.ground_truth_dir) if self.ground_truth_dir else None,
                credential_path=Path(self.credential_path) if self.credential_path else None,
            )
        else:
            fixture = Path(self.fixture_dir) if self.fixture_dir else Path(__file__).resolve().parents[2] / "fixtures/mini_orders"
            spec = TaskSpec.local(fixture)
        group = ELTGroupBuilder(spec, Path(self.run_root), self.model_name_for_tokenizer, self.group_size, self.max_turns)
        return ELTDataset([group]), None


async def _main(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    builder = ELTDatasetBuilder(
        model_name_for_tokenizer=args.model, run_root=str(run_root),
        official_repo=args.official_repo, task_id=args.task_id,
        inputs_dir=args.inputs_dir, ground_truth_dir=args.ground_truth_dir,
        credential_path=args.credential_path, fixture_dir=args.fixture_dir,
        group_size=args.group_size, max_turns=args.max_turns,
    )
    config = train.Config(
        model_name=args.model, recipe_name="elt_bench_rlvr",
        renderer_name=model_info.get_recommended_renderer_name(args.model),
        log_path=str(run_root / "tinker-log"), dataset_builder=builder,
        learning_rate=args.learning_rate, lora_rank=args.lora_rank,
        max_tokens=4096, eval_every=0, save_every=1, max_steps=args.max_steps,
    )
    await train.main(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ELT RLVR with Tinker cookbook")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--run-root", default="run_artifacts")
    parser.add_argument("--fixture-dir")
    parser.add_argument("--official-repo")
    parser.add_argument("--task-id", default="books")
    parser.add_argument("--inputs-dir")
    parser.add_argument("--ground-truth-dir")
    parser.add_argument("--credential-path")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
