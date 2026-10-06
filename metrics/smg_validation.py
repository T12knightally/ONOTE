"""Deterministic, non-aggregated structural checks for ONOTE SMG outputs.

These checks are audit fields only. They are not converted into the reported
1--5 GPT-5 structural-compliance score.
"""

from __future__ import annotations

import re
from fractions import Fraction


_ABC_TOKEN = re.compile(
    r"(?:\|\]|:\||\|:|::|\[\||\|)"
    r"|(?:\^\^|__|\^|_|=)?[A-Ga-gzZxX][,']*(?:\d+)?(?:/\d*)?"
)
_ABC_NOTE_PREFIX = re.compile(r"(?:\^\^|__|\^|_|=)?[A-Ga-gzZxX][,']*")
_JIANPU_FRACTION = re.compile(r"[_<>^]*[0-7]\((\d+)/(\d+)\)")
_JIANPU_LEGACY = re.compile(r"(?:[0-7]|-|\([0-7]\s+[0-7]\))")
_TAB_ROW = re.compile(r"^\s*([eBGDAE])\s*\|(.*)$", re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    """Remove one surrounding Markdown code fence, if present."""
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        value = "\n".join(lines).strip()
    return value


def extract_structural_score(judge_output: str | None) -> float | None:
    """Extract the judge's overall structural-compliance score, if present."""
    if not judge_output or "[Judge Error:" in judge_output:
        return None
    match = re.search(
        r"Structural\s+Compliance\s+Score\s*:\s*(\d+(?:\.\d+)?)\s*/\s*5",
        judge_output,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    score = float(match.group(1))
    return score if 1 <= score <= 5 else None


def extract_guitar_diagnostic_scores(judge_output: str | None) -> dict[str, float | None]:
    """Extract guitar layout and fret-span proxy scores without combining them."""
    output = judge_output or ""
    values: dict[str, float | None] = {}
    patterns = {
        "layout_score": r"Layout\s+Score\s*:\s*(\d+(?:\.\d+)?)\s*/\s*5",
        "fingering_constraint_score": r"Fingering(?:-Constraint)?\s+Score\s*:\s*(\d+(?:\.\d+)?)\s*/\s*5",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, output, flags=re.IGNORECASE)
        score = float(match.group(1)) if match else None
        values[field] = score if score is not None and 1 <= score <= 5 else None
    return values


def _split_measures(tokens: list[str]) -> list[list[str]]:
    measures: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in {"|", "|]", "[|", ":|", "|:", "::"}:
            if current:
                measures.append(current)
                current = []
        else:
            current.append(token)
    if current:
        measures.append(current)
    return measures


def _abc_duration(token: str, default_length: Fraction) -> Fraction:
    prefix = _ABC_NOTE_PREFIX.match(token)
    if prefix is None:
        raise ValueError(f"Invalid ABC note token: {token}")
    suffix = token[prefix.end():]
    if not suffix:
        multiplier = Fraction(1, 1)
    elif "/" not in suffix:
        multiplier = Fraction(int(suffix), 1)
    elif suffix.startswith("/"):
        denominator = int(suffix[1:]) if suffix[1:] else 2
        multiplier = Fraction(1, denominator)
    else:
        numerator, denominator = suffix.split("/", 1)
        multiplier = Fraction(int(numerator), int(denominator) if denominator else 2)
    return default_length * multiplier


def _validate_abc(text: str, expected_measures: int) -> dict[str, object]:
    value = strip_code_fence(text)
    headers: dict[str, str] = {}
    body_lines: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        match = re.match(r"^([A-Za-z]):\s*(.*)$", stripped)
        if match:
            headers[match.group(1).upper()] = match.group(2).strip()
        else:
            body_lines.append(stripped)

    required_headers = {"X", "T", "M", "L", "K"}
    header_ok = required_headers.issubset(headers)
    meter_match = re.fullmatch(r"(\d+)/(\d+)", headers.get("M", ""))
    length_match = re.fullmatch(r"(\d+)/(\d+)", headers.get("L", ""))
    if not meter_match or not length_match:
        return {
            "syntax_valid": False,
            "measure_count": None,
            "measure_count_ok": False,
            "meter_valid": False,
            "layout_valid": None,
            "fret_span_valid": None,
            "max_fret_span": None,
            "renderability_status": "not_assessed",
        }

    try:
        meter_target = Fraction(int(meter_match.group(1)), int(meter_match.group(2)))
        default_length = Fraction(int(length_match.group(1)), int(length_match.group(2)))
    except (ValueError, ZeroDivisionError):
        return {
            "syntax_valid": False,
            "measure_count": None,
            "measure_count_ok": False,
            "meter_valid": False,
            "layout_valid": None,
            "fret_span_valid": None,
            "max_fret_span": None,
            "renderability_status": "not_assessed",
        }
    body = " ".join(body_lines)
    tokens = [match.group(0) for match in _ABC_TOKEN.finditer(body)]
    residue = _ABC_TOKEN.sub("", body)
    syntax_ok = header_ok and bool(tokens) and not residue.strip()

    measures: list[list[str]] = []
    current: list[str] = []
    try:
        for token in tokens:
            if token in {"|", "|]", "[|", ":|", "|:", "::"}:
                if current:
                    measures.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            measures.append(current)
    except (ValueError, ZeroDivisionError):
        syntax_ok = False
        measures = []

    # Recompute measure totals so each bar is checked independently.
    measure_totals: list[Fraction] = []
    current_total = Fraction(0, 1)
    try:
        for token in tokens:
            if token in {"|", "|]", "[|", ":|", "|:", "::"}:
                if current_total:
                    measure_totals.append(current_total)
                    current_total = Fraction(0, 1)
            else:
                current_total += _abc_duration(token, default_length)
        if current_total:
            measure_totals.append(current_total)
    except (ValueError, ZeroDivisionError):
        measure_totals = []

    count = len(measures) if syntax_ok else None
    return {
        "syntax_valid": syntax_ok,
        "measure_count": count,
        "measure_count_ok": count == expected_measures,
        "meter_valid": bool(measure_totals) and all(total == meter_target for total in measure_totals),
        "layout_valid": None,
        "fret_span_valid": None,
        "max_fret_span": None,
        "renderability_status": "not_assessed",
    }


def _validate_jianpu(text: str, expected_measures: int) -> dict[str, object]:
    value = strip_code_fence(text)
    saw_bar = "|" in value
    measures = [part.split() for part in value.split("|") if part.strip()]

    fraction_mode = bool(measures) and all(
        _JIANPU_FRACTION.fullmatch(token)
        for measure in measures for token in measure
    )
    legacy_mode = bool(measures) and all(
        _JIANPU_LEGACY.fullmatch(token)
        for measure in measures for token in measure
    )
    syntax_ok = saw_bar and (fraction_mode or legacy_mode)
    if fraction_mode:
        try:
            durations = [
                (int(match.group(1)), int(match.group(2)))
                for measure in measures for token in measure
                if (match := _JIANPU_FRACTION.fullmatch(token))
            ]
            syntax_ok = syntax_ok and all(numerator > 0 and denominator > 0 for numerator, denominator in durations)
            meter_valid = syntax_ok and all(
                sum((Fraction(*map(int, _JIANPU_FRACTION.fullmatch(token).groups()))
                     for token in measure), Fraction(0, 1)) == 1
                for measure in measures
            )
        except (ValueError, ZeroDivisionError):
            syntax_ok = False
            meter_valid = False
    elif legacy_mode:
        # The appendix prompt defines each ordinary number, rest, or extension dash
        # as one beat; a parenthesized pair is two eighth notes, also one beat.
        meter_valid = all(len(measure) == 4 for measure in measures)
    else:
        meter_valid = False

    count = len(measures) if syntax_ok else None
    return {
        "syntax_valid": syntax_ok,
        "measure_count": count,
        "measure_count_ok": count == expected_measures,
        "meter_valid": meter_valid,
        "layout_valid": None,
        "fret_span_valid": None,
        "max_fret_span": None,
        "renderability_status": "not_assessed",
    }


def _validate_guitar_tab(text: str, expected_measures: int) -> dict[str, object]:
    value = strip_code_fence(text)
    required = ["E", "B", "G", "D", "A", "E"]
    # The first and sixth strings are both labelled E; accept common e/B/G/D/A/E order.
    ordered_labels = [match.group(1).upper() for line in value.splitlines()
                      if (match := _TAB_ROW.match(line))]
    row_payloads = [
        [part.strip() for part in match.group(2).split("|") if part.strip()]
        for line in value.splitlines() if (match := _TAB_ROW.match(line))
    ]
    row_labels_valid = len(ordered_labels) == 6 and ordered_labels == required
    measure_counts = {len(parts) for parts in row_payloads}
    consistent_measures = len(measure_counts) == 1 and bool(row_payloads)
    measure_count = next(iter(measure_counts)) if consistent_measures else None
    payload_syntax_valid = all(
        all(re.fullmatch(r"[-0-9]+", measure) for measure in parts)
        for parts in row_payloads
    )
    widths_valid = all(len(measure) == 16 for parts in row_payloads for measure in parts)
    layout_valid = row_labels_valid and consistent_measures and payload_syntax_valid and widths_valid

    spans: list[int] = []
    if row_labels_valid and consistent_measures and payload_syntax_valid and widths_valid:
        for measure_index in range(measure_count or 0):
            for column in range(16):
                frets = [
                    int(row_payloads[row_index][measure_index][column])
                    for row_index in range(6)
                    if row_payloads[row_index][measure_index][column].isdigit()
                ]
                if len(frets) > 1:
                    spans.append(max(frets) - min(frets))

    max_span = max(spans) if spans else None
    return {
        "syntax_valid": row_labels_valid and payload_syntax_valid,
        "measure_count": measure_count,
        "measure_count_ok": measure_count == expected_measures,
        "meter_valid": widths_valid and consistent_measures,
        "layout_valid": layout_valid,
        "fret_span_valid": max_span is None or max_span <= 7,
        "max_fret_span": max_span,
        "renderability_status": "not_assessed",
    }


def validate_smg_output(task: str, text: str, expected_measures: int) -> dict[str, object]:
    """Run separate deterministic checks; never aggregate them into an SMG score."""
    if task == "staff":
        result = _validate_abc(text, expected_measures)
    elif task == "jianpu":
        result = _validate_jianpu(text, expected_measures)
    elif task == "guitar":
        result = _validate_guitar_tab(text, expected_measures)
    else:
        raise ValueError(f"Unsupported SMG task: {task}")
    return {f"deterministic_{key}": value for key, value in result.items()}
