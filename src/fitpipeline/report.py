"""Human-readable console report of a list of prediction entries."""

from __future__ import annotations

from collections import Counter

_GREEN, _YELLOW, _DIM, _RESET = "\033[32m", "\033[33m", "\033[2m", "\033[0m"


def _paint(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{_RESET}" if enabled else text


def _format_parameters(parameters: dict) -> str:
    def number(value):
        return f"{value:.6g}"

    return ", ".join(
        f"{key}=[{', '.join(number(v) for v in value)}]"
        if isinstance(value, list)
        else f"{key}={number(value)}"
        for key, value in parameters.items()
    )


def format_report(predictions: list[dict], *, verbose: bool = False, color: bool = False) -> str:
    """One table row per entry; with `verbose`, also the reason and fitted parameters."""
    header = (
        f"{'mesh':<8} {'seg':>3}  {'result':<11} {'conf':>5}  {'fit':<7} "
        f"{'vertex RMS':>10}  {'normal RMS':>10}"
    )
    lines = [header, "-" * len(header)]
    for entry in predictions:
        fit = entry["fit"]
        residual = fit.get("residual")
        result = entry["primitive_type"] or "unresolved"
        row = (
            f"{entry['mesh_id']:<8} {entry['segment_id']:>3}  "
            f"{_paint(f'{result:<11}', _GREEN if entry['kind'] == 'primitive' else _YELLOW, color)} "
            f"{entry['confidence']:>5.2f}  {fit['status']:<7} "
            + (
                f"{residual['relative_vertex_rms']:>10.1e}  "
                f"{residual['facet_normal_rms_degrees']:>9.2f}°"
                if residual
                else f"{'-':>10}  {'-':>10}"
            )
        )
        lines.append(row)
        if verbose:
            lines.append(_paint(f"           reason: {entry['reason']}", _DIM, color))
            if fit["parameters"]:
                lines.append(_paint(f"           {_format_parameters(fit['parameters'])}", _DIM, color))
    return "\n".join(lines)


def format_summary(predictions: list[dict]) -> str:
    counts = Counter(p["primitive_type"] or "unresolved" for p in predictions)
    return f"{len(predictions)} segments: " + ", ".join(
        f"{name} {count}" for name, count in sorted(counts.items())
    )
