"""LLM/tool-calling eval harness (docs/tz/06_llm.md §3.8).

    uv run python scripts/eval_llm_tools.py
    uv run python scripts/eval_llm_tools.py --model qwen3:8b --model qwen2.5:7b
    uv run python scripts/eval_llm_tools.py --fastpath off

Every tool call runs against a *mock* registry: real ``@tool`` functions are
never invoked, only recorded (name + args), so this is safe to run against
any future dangerous tool without side effects. That means metrics can only
judge whether the agent picked the right tool/arguments, not whether a tool's
own logic is correct (that's ``test_tool_schema.py``'s job) -- the LLM's
follow-up reply is generated from a mocked (not real) tool result, so
``must_contain`` checks in the dataset should target the model's own
conversational phrasing, not a computed value it was never actually given.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from loguru import logger

from ultron.config import ConfigError, RootConfig, load_config
from ultron.llm.agent import Agent, DoneEvent, ErrorEvent, SentenceEvent, ToolStartEvent
from ultron.llm.client import OllamaClient
from ultron.llm.fastpath import FastPath
from ultron.tools import default_registry
from ultron.tools.base import Tool, ToolRegistry, ToolResult
from ultron.tools.demo_tools import ExpressionError, evaluate_expression

_DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "llm_eval_uz.yaml"
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_ENGLISH_WORDS = {
    "the",
    "is",
    "are",
    "you",
    "and",
    "hello",
    "hi",
    "please",
    "what",
    "how",
    "time",
    "date",
    "result",
    "sorry",
    "yes",
    "no",
    "ok",
    "okay",
    "thanks",
}


@dataclass
class EvalCase:
    input: str
    expected_tool: str | None
    expected_args: dict[str, Any] = field(default_factory=dict)
    must_contain: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case: EvalCase
    called_tool: str | None
    called_args: dict[str, Any]
    response_text: str
    first_sentence_latency_ms: float | None


class _AutoConfirm:
    async def confirm(self, summary_uz: str) -> bool:
        return True


def _load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    cases = []
    for item in raw:
        must_contain = item.get("must_contain") or []
        if isinstance(must_contain, str):
            must_contain = [must_contain]
        cases.append(
            EvalCase(
                input=item["input"],
                expected_tool=item.get("expected_tool"),
                expected_args=item.get("expected_args") or {},
                must_contain=must_contain,
            )
        )
    return cases


def _build_mock_registry(source: ToolRegistry, calls: list[dict[str, Any]]) -> ToolRegistry:
    """Same names/schemas/dangerous flags as ``source``, but every call is
    recorded into ``calls`` instead of running the real tool body."""
    mock = ToolRegistry(confirmation=_AutoConfirm())
    for name, src_tool in source.all_tools().items():

        def _make_mock_func(tool_name: str) -> Any:
            async def _mock_func(**kwargs: Any) -> ToolResult:
                calls.append({"name": tool_name, "args": kwargs})
                return ToolResult(ok=True, data=kwargs)

            return _mock_func

        mock.register(
            Tool(
                name=name,
                description=src_tool.description,
                dangerous=src_tool.dangerous,
                func=_make_mock_func(name),
                schema=src_tool.schema,
                arg_model=src_tool.arg_model,
            )
        )
    return mock


def _values_match(expected: Any, actual: Any) -> bool:
    """Partial/semantic match: arithmetic-looking strings compare by
    evaluated value (so "17 * 23" == "17*23" == "23*17"); everything else,
    exact equality."""
    if isinstance(expected, str) and isinstance(actual, str):
        try:
            return evaluate_expression(expected) == evaluate_expression(actual)
        except (ExpressionError, ZeroDivisionError, SyntaxError, TypeError):
            return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def _tool_correct(result: CaseResult) -> bool:
    if result.case.expected_tool is None:
        return result.called_tool is None
    return result.called_tool == result.case.expected_tool


def _args_correct(result: CaseResult) -> bool | None:
    """``None`` when not applicable (no tool expected, or nothing to check)."""
    if result.case.expected_tool is None or not result.case.expected_args:
        return None
    if result.called_tool != result.case.expected_tool:
        return False
    return all(
        _values_match(expected, result.called_args.get(key))
        for key, expected in result.case.expected_args.items()
    )


def _must_contain_ok(result: CaseResult) -> bool:
    if not result.case.must_contain:
        return True
    text = result.response_text.lower()
    return all(token.lower() in text for token in result.case.must_contain)


def _foreign_word_ratio(text: str) -> float:
    """Simple heuristic (docs/tz/06_llm.md §3.8): share of words that are
    Cyrillic or a common English filler word."""
    words = re.findall(r"[^\s.,!?;:()\"']+", text.lower())
    if not words:
        return 0.0
    foreign = sum(1 for w in words if _CYRILLIC_RE.search(w) or w in _ENGLISH_WORDS)
    return foreign / len(words)


async def _run_case(agent: Agent, calls: list[dict[str, Any]], case: EvalCase) -> CaseResult:
    calls.clear()
    agent.reset_history()
    sentences: list[str] = []
    first_latency_ms: float | None = None
    start = time.monotonic()

    async for event in agent.respond(case.input):
        if isinstance(event, SentenceEvent):
            if first_latency_ms is None:
                first_latency_ms = (time.monotonic() - start) * 1000
            sentences.append(event.text)
        elif isinstance(event, ErrorEvent):
            if first_latency_ms is None:
                first_latency_ms = (time.monotonic() - start) * 1000
            sentences.append(event.message)
        elif isinstance(event, ToolStartEvent | DoneEvent):
            pass

    called = calls[0] if calls else None
    return CaseResult(
        case=case,
        called_tool=called["name"] if called else None,
        called_args=called["args"] if called else {},
        response_text=" ".join(sentences),
        first_sentence_latency_ms=first_latency_ms,
    )


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


async def _eval_model(
    config: RootConfig, model: str, cases: list[EvalCase], *, fastpath_enabled: bool
) -> dict[str, Any]:
    llm_config = config.llm.model_copy(update={"model": model})
    client = OllamaClient(llm_config)
    if not await client.health():
        logger.warning("Model '{}' uchun Ollama/model tayyor emas, baribir davom etiladi", model)
    else:
        await client.ensure_loaded()

    calls: list[dict[str, Any]] = []
    mock_registry = _build_mock_registry(default_registry, calls)
    fastpath = FastPath(mock_registry)

    results: list[CaseResult] = []
    for case in cases:
        agent = Agent(
            client,
            mock_registry,
            fastpath,
            llm_config,
            city=config.tools.weather.city,
            fastpath_enabled=fastpath_enabled,
        )
        results.append(await _run_case(agent, calls, case))

    tool_accuracy = sum(_tool_correct(r) for r in results) / len(results)

    arg_checks = [c for c in (_args_correct(r) for r in results) if c is not None]
    arg_accuracy = (sum(arg_checks) / len(arg_checks)) if arg_checks else None

    responded = [r for r in results if r.response_text.strip()]
    uzbek_ratio = (
        sum(1 for r in responded if _foreign_word_ratio(r.response_text) == 0.0) / len(responded)
        if responded
        else None
    )

    checkable = [r for r in results if r.case.must_contain]
    must_contain_ratio = (
        (sum(_must_contain_ok(r) for r in checkable) / len(checkable)) if checkable else None
    )

    latencies = [
        r.first_sentence_latency_ms for r in results if r.first_sentence_latency_ms is not None
    ]
    p50 = statistics.median(latencies) if latencies else None
    p95 = _percentile95(latencies)

    # docs/tz/06_llm.md §5's latency target is explicitly "LLM yo'li,
    # tool'siz" (one round-trip) -- a tool case takes two LLM calls (decide
    # to call the tool, then speak the result), so mixing it into the
    # headline p50/p95 would measure something the TZ never claimed.
    no_tool_latencies = [
        r.first_sentence_latency_ms
        for r in results
        if r.case.expected_tool is None and r.first_sentence_latency_ms is not None
    ]
    no_tool_p50 = statistics.median(no_tool_latencies) if no_tool_latencies else None
    no_tool_p95 = _percentile95(no_tool_latencies)

    return {
        "model": model,
        "n_cases": len(results),
        "tool_accuracy": tool_accuracy,
        "arg_accuracy": arg_accuracy,
        "uzbek_ratio": uzbek_ratio,
        "no_tool_p50_ms": no_tool_p50,
        "no_tool_p95_ms": no_tool_p95,
        "must_contain_ratio": must_contain_ratio,
        "p50_ms": p50,
        "p95_ms": p95,
        "results": results,
    }


def _fmt_pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "n/a"


def _fmt_ms(value: float | None) -> str:
    return f"{value:.0f} ms" if value is not None else "n/a"


def _write_report(config: RootConfig, reports: list[dict[str, Any]]) -> Path:
    out_dir = Path(config.app.data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"llm_eval_{date.today():%Y%m%d}.md"

    lines = [f"# LLM/tool-calling eval — {date.today():%Y-%m-%d}", ""]
    lines.append(
        "| Model | N | Tool aniqlik | Argument aniqlik | O'zbekcha | must_contain | p50 | p95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in reports:
        lines.append(
            f"| {r['model']} | {r['n_cases']} | {_fmt_pct(r['tool_accuracy'])} "
            f"| {_fmt_pct(r['arg_accuracy'])} | {_fmt_pct(r['uzbek_ratio'])} "
            f"| {_fmt_pct(r['must_contain_ratio'])} "
            f"| {_fmt_ms(r['p50_ms'])} | {_fmt_ms(r['p95_ms'])} |"
        )
    lines.append("")
    lines.append(
        "p50/p95 above are across *all* cases (mixes 1-round-trip conversational turns"
        " with 2-round-trip tool calls). docs/tz/06_llm.md §5's latency target is"
        " explicitly for the tool-free path; that subset alone:"
    )
    lines.append("")
    lines.append("| Model | p50 (tool'siz) | p95 (tool'siz) |")
    lines.append("|---|---|---|")
    for r in reports:
        lines.append(
            f"| {r['model']} | {_fmt_ms(r['no_tool_p50_ms'])} | {_fmt_ms(r['no_tool_p95_ms'])} |"
        )
    lines.append("")

    for r in reports:
        lines.append(f"## {r['model']} — muvaffaqiyatsiz holatlar")
        failures = [
            res
            for res in r["results"]
            if not _tool_correct(res) or _args_correct(res) is False or not _must_contain_ok(res)
        ]
        if not failures:
            lines.append("Hammasi o'tdi.")
        for res in failures:
            lines.append(
                f"- input: `{res.case.input}` | expected_tool={res.case.expected_tool} "
                f"expected_args={res.case.expected_args} "
                f"| got_tool={res.called_tool} got_args={res.called_args} "
                f"| response={res.response_text!r}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron LLM/tool-calling eval")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Solishtiriladigan model (bir necha marta berish mumkin)",
    )
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA_PATH)
    parser.add_argument("--fastpath", choices=["on", "off"], default="on")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)
    try:
        config = load_config()
    except ConfigError as exc:
        logger.info("Config yuklab bo'lmadi:\n{}", exc)
        return 1

    cases = _load_cases(args.data)
    if len(cases) < 40:
        logger.warning("Eval to'plamida {} ta holat bor, TZ kamida 40 ta talab qiladi", len(cases))
    else:
        logger.info("{} ta holat yuklandi ({})", len(cases), args.data)

    models = args.models or [config.llm.model]
    fastpath_enabled = args.fastpath == "on"

    async def _run_all() -> list[dict[str, Any]]:
        return [
            await _eval_model(config, model, cases, fastpath_enabled=fastpath_enabled)
            for model in models
        ]

    reports = asyncio.run(_run_all())
    out_path = _write_report(config, reports)
    logger.info("Hisobot yozildi: {}", out_path)

    for r in reports:
        logger.info(
            "{}: tool_acc={} args_acc={} uzbek={} must_contain={} p50={} p95={} "
            "| tool'siz p50={} p95={}",
            r["model"],
            _fmt_pct(r["tool_accuracy"]),
            _fmt_pct(r["arg_accuracy"]),
            _fmt_pct(r["uzbek_ratio"]),
            _fmt_pct(r["must_contain_ratio"]),
            _fmt_ms(r["p50_ms"]),
            _fmt_ms(r["p95_ms"]),
            _fmt_ms(r["no_tool_p50_ms"]),
            _fmt_ms(r["no_tool_p95_ms"]),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
