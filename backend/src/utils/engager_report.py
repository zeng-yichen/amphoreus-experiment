"""Engager Report — Phase 1 (LinkedIn CSV-based).

Classifies LinkedIn post engagers from a native CSV export into four buckets,
updates a per-client history store, and generates a two-chart PNG.

Classification buckets
----------------------
icp_match  — matches the client's ICP, with a specific segment label.
internal   — part of the client's own team.
orbit      — previously seen but does not match ICP this run.
non_icp    — everything else.

Classification is fully LLM-driven — no hard thresholds, no keyword lists.
The model receives the ICP description, segment descriptions, client company
name, and previously-seen engager names, then reasons about each profile.
The chart and this module are analyst inputs, not automated decisions.

Data flow
---------
1. parse_csv()           → normalise LinkedIn export into list[EngagerRow]
2. classify_engagers()   → one LLM call per batch, returns ClassifiedEngager list
3. update_history()      → upsert into backend/data/{company}/engagers.json
4. generate_charts()     → return matplotlib Figure (two subplots)
5. save_report()         → write PNG to disk

ICP segment definition (optional)
-----------------------------------
Add a ``segments`` list to ``memory/{company}/icp_definition.json``::

    {
      "description": "...",
      "anti_description": "...",
      "segments": [
        {"label": "AI GTM/Ecosystem",  "description": "GTM, ecosystem, partnerships, or devrel roles at AI companies"},
        {"label": "AI Inference",       "description": "Engineers working on LLM inference, model serving, or GPU infrastructure"}
      ]
    }

If ``segments`` is absent, all ICP matches fall into "ICP Match".
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
COLOR_ICP      = "#3DBA78"   # accent green
COLOR_NON_ICP  = "#B0B8C4"   # muted gray
COLOR_INTERNAL = "#7EC8E3"   # light blue
COLOR_ORBIT    = "#B39DDB"   # muted purple
COLOR_BG       = "#FFFFFF"

# ── Data paths ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]          # backend/src/utils/ → project root


def _engager_store_path(company: str) -> Path:
    return _PROJECT_ROOT / "backend" / "data" / company / "engagers.json"


def _icp_definition_path(company: str) -> Path:
    return _PROJECT_ROOT / "memory" / company / "icp_definition.json"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CSV PARSING
# ══════════════════════════════════════════════════════════════════════════════

_COL_ALIASES: dict[str, str] = {
    "name": "name",
    "member name": "name",
    "commenter name": "name",
    "full name": "name",
    "headline": "headline",
    "member headline": "headline",
    "commenter headline": "headline",
    "title": "headline",
    "job title": "headline",
    "company": "company",
    "member company": "company",
    "organization": "company",
    "current company": "company",
    "employer": "company",
    "reaction type": "interaction",
    "reaction": "interaction",
    "comment": "interaction",
    "interaction": "interaction",
    "activity": "interaction",
    "post id": "post_id",
    "post url": "post_id",
    "content id": "post_id",
    "update urn": "post_id",
    "date": "date",
    "reaction date": "date",
    "comment date": "date",
}


class EngagerRow:
    """Single normalised engager record from the CSV."""

    __slots__ = ("name", "headline", "company", "interaction", "post_id", "raw_date")

    def __init__(self, name: str, headline: str, company: str,
                 interaction: str, post_id: str, raw_date: str):
        self.name = name.strip()
        self.headline = headline.strip()
        self.company = company.strip()
        self.interaction = interaction.strip()
        self.post_id = post_id.strip()
        self.raw_date = raw_date.strip()

    def __repr__(self) -> str:
        return f"<EngagerRow name={self.name!r} headline={self.headline!r} company={self.company!r}>"


def parse_csv(csv_path: str | Path, window_days: int = 0) -> list[EngagerRow]:
    """Parse a LinkedIn analytics export CSV and return normalised rows.

    Handles both reaction and comment exports. Unknown columns are ignored.
    Rows missing a name are skipped.

    Args:
        csv_path:    Path to the CSV file.
        window_days: If > 0, only keep rows within this many days of today.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    cutoff: Optional[date] = None
    if window_days > 0:
        cutoff = date.today() - timedelta(days=window_days)

    rows: list[EngagerRow] = []

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            logger.warning("[engager_report] CSV has no headers: %s", path)
            return rows

        col_map: dict[str, str] = {}
        for orig in reader.fieldnames:
            canonical = _COL_ALIASES.get(orig.lower().strip())
            if canonical:
                col_map[orig] = canonical

        logger.info(
            "[engager_report] Detected columns: %s → canonical: %s",
            list(reader.fieldnames), col_map,
        )

        def _get(row: dict, field: str) -> str:
            for orig, canon in col_map.items():
                if canon == field and orig in row:
                    return (row[orig] or "").strip()
            return ""

        for raw in reader:
            name = _get(raw, "name")
            if not name:
                continue

            raw_date = _get(raw, "date")
            if cutoff and raw_date:
                try:
                    row_date = _parse_date(raw_date)
                    if row_date < cutoff:
                        continue
                except ValueError:
                    pass

            rows.append(EngagerRow(
                name=name,
                headline=_get(raw, "headline"),
                company=_get(raw, "company"),
                interaction=_get(raw, "interaction"),
                post_id=_get(raw, "post_id"),
                raw_date=raw_date,
            ))

    logger.info("[engager_report] Parsed %d rows from %s", len(rows), path.name)
    return rows


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ICP DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

def _load_icp_definition(company: str) -> dict:
    path = _icp_definition_path(company)
    if not path.exists():
        logger.warning("[engager_report] No ICP definition found for %s", company)
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[engager_report] Failed to load ICP definition: %s", e)
        return {}


def _load_client_company_name(company: str) -> str:
    """Derive the client's own company identifier for the LLM context.

    Check order:
    1. profile.json ``company`` or ``organization`` field
    2. First token of the client slug (e.g. "hume" from "hume-andrew")
    """
    profile_path = _PROJECT_ROOT / "memory" / company / "profile.json"
    if profile_path.exists():
        try:
            with profile_path.open(encoding="utf-8") as f:
                data = json.load(f)
            name = (data.get("company") or data.get("organization") or "").strip()
            if name:
                return name
        except Exception:
            pass
    return company.split("-")[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LLM CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class ClassifiedEngager:
    """Engager row with classification metadata attached."""

    __slots__ = ("row", "classification", "segment")

    def __init__(self, row: EngagerRow, classification: str, segment: Optional[str]):
        self.row = row
        self.classification = classification   # icp_match | internal | orbit | non_icp
        self.segment = segment                 # e.g. "AI GTM/Ecosystem" or None


def _build_classification_prompt(
    rows: list[EngagerRow],
    icp_def: dict,
    client_company: str,
    known_names: set[str],
) -> str:
    """Construct the full classification prompt for a batch of engagers."""
    description = icp_def.get("description", "")
    anti_description = icp_def.get("anti_description", "")
    segments: list[dict] = icp_def.get("segments", [])

    lines: list[str] = []

    lines.append("You are classifying LinkedIn post engagers for an analyst reviewing audience quality.")
    lines.append("")
    lines.append(f"CLIENT COMPANY: {client_company}")
    lines.append("")

    if description:
        lines.append("ICP — WHO WE WANT ENGAGING:")
        lines.append(description)
        lines.append("")

    if anti_description:
        lines.append("ANTI-ICP — WHO WE DO NOT WANT:")
        lines.append(anti_description)
        lines.append("")

    if segments:
        lines.append("ICP SEGMENTS (assign one to each icp_match engager):")
        for seg in segments:
            label = seg.get("label", "")
            desc = seg.get("description", "")
            lines.append(f"  - {label}: {desc}")
        lines.append("")
    else:
        lines.append('ICP SEGMENT: use "ICP Match" for all icp_match engagers.')
        lines.append("")

    if known_names:
        lines.append("PREVIOUSLY SEEN ENGAGERS (eligible for 'orbit' classification):")
        lines.append(", ".join(sorted(known_names)[:80]))
        lines.append("")

    lines.append("CLASSIFICATION RULES:")
    lines.append("  icp_match  — profile fits the ICP description above")
    lines.append(f"  internal   — person works at {client_company} or is clearly part of their team")
    lines.append("  orbit      — name is in the previously-seen list AND does not fit the ICP")
    lines.append("  non_icp    — everyone else")
    lines.append("")
    lines.append("Priority: internal > icp_match > orbit > non_icp")
    lines.append("")
    lines.append("ENGAGER PROFILES:")
    for i, row in enumerate(rows):
        parts = []
        if row.name:
            parts.append(f"name: {row.name}")
        if row.headline:
            parts.append(f"headline: {row.headline}")
        if row.company:
            parts.append(f"company: {row.company}")
        lines.append(f"{i + 1}. {' | '.join(parts) if parts else '(no data)'}")

    lines.append("")
    lines.append("For each numbered engager, output EXACTLY one line:")
    lines.append("  N: classification | segment_label")
    lines.append("")
    lines.append("Rules:")
    lines.append("  - For icp_match: include the segment label after the pipe")
    lines.append('  - For all others: use "-" after the pipe')
    lines.append("  - Output ONLY the numbered lines, nothing else")
    lines.append("")
    lines.append(f"Example output for 3 engagers:")
    lines.append("  1: icp_match | Voice AI Builders")
    lines.append("  2: non_icp | -")
    lines.append("  3: internal | -")

    return "\n".join(lines)


def _parse_classification_output(raw: str, rows: list[EngagerRow], known_names: set[str]) -> list[ClassifiedEngager]:
    """Parse LLM output into ClassifiedEngager objects.

    Falls back to non_icp / orbit for any line that cannot be parsed.
    """
    valid_classes = {"icp_match", "internal", "orbit", "non_icp"}
    result_map: dict[int, ClassifiedEngager] = {}

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: "N: classification | segment" or "N: classification"
        m = re.match(r"^(\d+)\s*[:)]\s*(\w+)\s*(?:\|\s*(.+))?", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        cls = m.group(2).lower().strip()
        seg_raw = (m.group(3) or "").strip()
        segment = seg_raw if seg_raw and seg_raw != "-" else None

        if cls not in valid_classes:
            cls = "non_icp"
        if idx < 0 or idx >= len(rows):
            continue

        result_map[idx] = ClassifiedEngager(rows[idx], cls, segment)

    # Fill in any rows the LLM missed
    name_norm = lambda n: n.strip().lower()
    results: list[ClassifiedEngager] = []
    for i, row in enumerate(rows):
        if i in result_map:
            results.append(result_map[i])
        else:
            # Conservative fallback
            fallback_cls = "orbit" if name_norm(row.name) in known_names else "non_icp"
            results.append(ClassifiedEngager(row, fallback_cls, None))

    return results


def classify_engagers(
    rows: list[EngagerRow],
    company: str,
    known_names: set[str],
) -> list[ClassifiedEngager]:
    """Classify every engager in a single LLM call per batch.

    The model receives the full ICP text, segment descriptions, client company
    name, and previously-seen engager names.  No hard thresholds or keyword
    matching — all reasoning is delegated to the model.

    Args:
        rows:         Parsed CSV rows.
        company:      Client slug (e.g. "hume-andrew").
        known_names:  Normalised names from the existing engagers.json store.

    Returns:
        List of :class:`ClassifiedEngager`.
    """
    if not rows:
        return []

    icp_def = _load_icp_definition(company)
    client_co = _load_client_company_name(company)

    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        logger.warning("[engager_report] anthropic not installed — all classified as non_icp")
        return [ClassifiedEngager(r, "non_icp", None) for r in rows]

    all_results: list[ClassifiedEngager] = []
    chunk_size = 50

    for chunk_start in range(0, len(rows), chunk_size):
        chunk = rows[chunk_start : chunk_start + chunk_size]
        prompt = _build_classification_prompt(chunk, icp_def, client_co, known_names)

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            chunk_results = _parse_classification_output(raw, chunk, known_names)
        except Exception as e:
            logger.warning("[engager_report] LLM classification failed for chunk: %s", e)
            name_norm = lambda n: n.strip().lower()
            chunk_results = [
                ClassifiedEngager(r, "orbit" if name_norm(r.name) in known_names else "non_icp", None)
                for r in chunk
            ]

        all_results.extend(chunk_results)

    counts = {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0}
    for ce in all_results:
        counts[ce.classification] = counts.get(ce.classification, 0) + 1

    logger.info("[engager_report] Classified %d rows: %s", len(all_results), counts)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HISTORY STORE
# ══════════════════════════════════════════════════════════════════════════════

def load_history(company: str) -> dict:
    """Load backend/data/{company}/engagers.json — returns empty store if absent."""
    path = _engager_store_path(company)
    if not path.exists():
        return {"engagers": []}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if "engagers" not in data:
            data["engagers"] = []
        return data
    except Exception as e:
        logger.warning("[engager_report] Failed to load history: %s", e)
        return {"engagers": []}


def history_names(store: dict) -> set[str]:
    """Return normalised names from the history store for orbit detection."""
    return {e["name"].strip().lower() for e in store.get("engagers", []) if e.get("name")}


def update_history(
    company: str,
    classified: list[ClassifiedEngager],
    post_ids: list[str],
) -> dict:
    """Upsert classified engagers into the history store and save.

    Matches on normalised name. Updates ``last_seen``, ``segment``,
    ``classification``, and ``post_ids``.

    Returns the updated store dict.
    """
    store = load_history(company)
    today = date.today().isoformat()

    index: dict[str, dict] = {}
    for entry in store["engagers"]:
        key = entry.get("name", "").strip().lower()
        if key:
            index[key] = entry

    for ce in classified:
        key = ce.row.name.strip().lower()
        if not key:
            continue

        row_post_ids = [ce.row.post_id] if ce.row.post_id else []
        if post_ids:
            row_post_ids = list(set(row_post_ids + post_ids))

        if key in index:
            entry = index[key]
            entry["last_seen"] = today
            entry["classification"] = ce.classification
            if ce.segment:
                entry["segment"] = ce.segment
            if ce.row.headline and not entry.get("headline"):
                entry["headline"] = ce.row.headline
            if ce.row.company and not entry.get("company"):
                entry["company"] = ce.row.company
            existing_posts = set(entry.get("post_ids", []))
            existing_posts.update(row_post_ids)
            entry["post_ids"] = sorted(existing_posts)
        else:
            new_entry: dict = {
                "name": ce.row.name,
                "company": ce.row.company,
                "headline": ce.row.headline,
                "segment": ce.segment or "",
                "classification": ce.classification,
                "first_seen": today,
                "last_seen": today,
                "post_ids": sorted(set(row_post_ids)),
            }
            store["engagers"].append(new_entry)
            index[key] = new_entry

    path = _engager_store_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)
    logger.info("[engager_report] History updated → %d total engagers", len(store["engagers"]))
    return store


def engager_distribution(company: str) -> dict:
    """Load the current history store and return classification counts.

    Returns::

        {
            "icp_match": int,
            "non_icp": int,
            "internal": int,
            "orbit": int,
            "total": int,
            "segments": {"Segment Name": int, ...}
        }

    Useful for embedding in the progress report without a full CSV run.
    """
    store = load_history(company)
    counts: dict[str, int] = {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0}
    segments: dict[str, int] = {}
    for e in store.get("engagers", []):
        cls = e.get("classification", "non_icp")
        counts[cls] = counts.get(cls, 0) + 1
        if cls == "icp_match" and e.get("segment"):
            seg = e["segment"]
            segments[seg] = segments.get(seg, 0) + 1
    return {**counts, "total": sum(counts.values()), "segments": segments}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PIE SVG (pre-computed, embeddable in HTML reports)
# ══════════════════════════════════════════════════════════════════════════════

def render_engager_pie_svg(dist: dict, width: int = 300, height: int = 260) -> str:
    """Render an engager distribution pie chart as an inline SVG string.

    Pre-computes all arc math in Python so the HTML template just embeds the
    result directly — no LLM trig required.

    Args:
        dist:   Output of :func:`engager_distribution`.
        width:  SVG viewport width in px.
        height: SVG viewport height in px.

    Returns:
        A self-contained ``<svg>...</svg>`` string.
    """
    import math

    slices = [
        ("ICP Match", dist.get("icp_match", 0), COLOR_ICP),
        ("Non-ICP",   dist.get("non_icp",   0), COLOR_NON_ICP),
        ("Internal",  dist.get("internal",  0), COLOR_INTERNAL),
        ("Orbit",     dist.get("orbit",     0), COLOR_ORBIT),
    ]
    slices = [(l, v, c) for l, v, c in slices if v > 0]
    total = sum(v for _, v, _ in slices)
    if total == 0:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="{width//2}" y="{height//2}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="#aaa">No data</text></svg>'
        )

    cx, cy, r = width // 2, (height - 60) // 2 + 10, min(width, height - 60) // 2 - 16
    legend_y_start = cy + r + 18

    paths: list[str] = []
    labels: list[str] = []
    legend_items: list[str] = []

    angle = -math.pi / 2   # start at top

    for label, value, color in slices:
        sweep = 2 * math.pi * value / total
        end_angle = angle + sweep

        x1 = cx + r * math.cos(angle)
        y1 = cy + r * math.sin(angle)
        x2 = cx + r * math.cos(end_angle)
        y2 = cy + r * math.sin(end_angle)
        large_arc = 1 if sweep > math.pi else 0

        # Wedge path
        paths.append(
            f'<path d="M {cx} {cy} L {x1:.2f} {y1:.2f} '
            f'A {r} {r} 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z" '
            f'fill="{color}" stroke="white" stroke-width="1.5"/>'
        )

        # Percentage label at midpoint of arc (skip tiny slices)
        if sweep > 0.25:
            mid_angle = angle + sweep / 2
            lx = cx + (r * 0.65) * math.cos(mid_angle)
            ly = cy + (r * 0.65) * math.sin(mid_angle)
            pct = f"{100 * value / total:.0f}%"
            labels.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                f'dominant-baseline="middle" font-family="sans-serif" '
                f'font-size="11" font-weight="600" fill="white">{pct}</text>'
            )

        angle = end_angle

    # Legend (two-column, below pie)
    legend_cols = 2
    item_w = width // legend_cols
    for i, (label, value, color) in enumerate(slices):
        col = i % legend_cols
        row = i // legend_cols
        lx = col * item_w + 12
        ly = legend_y_start + row * 20
        pct = f"{100 * value / total:.0f}%"
        legend_items.append(
            f'<rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text x="{lx + 14}" y="{ly + 9}" font-family="sans-serif" '
            f'font-size="11" fill="#444">{label} ({value}, {pct})</text>'
        )

    inner = "\n  ".join(paths + labels + legend_items)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n  '
        f'{inner}\n</svg>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LEARNING SYSTEM BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

def bridge_to_ruan_mei(
    company: str,
    classified: list[ClassifiedEngager],
    ordinal_post_id: str,
    linkedin_post_url: str = "",
) -> bool:
    """Write CSV-derived ICP signal into RuanMei.

    Computes ``icp_match_rate`` from the classified engager list (internal
    engagers excluded — they're team members, not audience signal), then
    calls ``RuanMei.update_icp_reward()`` so the signal reaches the reward
    field immediately rather than waiting for the next hourly sync.

    The CSV-based signal is generally richer than the automated Ordinal
    scorer: it covers the full engager list (not capped at 30 reactions),
    uses per-person named-segment classification, and is human-verified.
    Calling this replaces any prior icp_reward on that observation.

    Args:
        company:          Client slug.
        classified:       Output of :func:`classify_engagers`.
        ordinal_post_id:  The Ordinal post UUID to update (matches
                          ``ordinal_post_id`` on the RuanMei observation).
        linkedin_post_url: Optional — stored on the observation if provided.

    Returns:
        True if the RuanMei observation was found and updated, False otherwise.
    """
    if not ordinal_post_id:
        logger.warning("[engager_report] bridge_to_ruan_mei called with no post ID — skipped")
        return False

    counts: dict[str, int] = {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0}
    for ce in classified:
        counts[ce.classification] = counts.get(ce.classification, 0) + 1

    # Internal engagers are excluded from the rate denominator.
    # They tell you nothing about whether the post attracted the right external audience.
    signal_total = counts["icp_match"] + counts["non_icp"] + counts["orbit"]
    if signal_total == 0:
        logger.warning("[engager_report] No non-internal engagers to derive ICP rate from — bridge skipped")
        return False

    icp_match_rate = counts["icp_match"] / signal_total

    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        updated = rm.update_icp_reward(
            ordinal_post_id,
            icp_score=icp_match_rate,
            linkedin_post_url=linkedin_post_url,
            icp_match_rate=icp_match_rate,
        )
    except Exception as e:
        logger.error("[engager_report] RuanMei bridge failed: %s", e)
        return False

    if not updated:
        logger.warning(
            "[engager_report] No scored observation found for ordinal_post_id=%r in %s",
            ordinal_post_id, company,
        )
        return False

    logger.info(
        "[engager_report] RuanMei updated for %s post %s…: icp_match_rate=%.1f%% (%d/%d engagers)",
        company, ordinal_post_id[:12],
        icp_match_rate * 100,
        counts["icp_match"], signal_total,
    )

    # (LOLA reward field propagation removed — content intelligence now
    # handled by RuanMei.recommend_context())

    return True


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MATPLOTLIB CHARTS (for standalone PNG output)
# ══════════════════════════════════════════════════════════════════════════════

def generate_charts(
    classified: list[ClassifiedEngager],
    company: str,
    title_suffix: str = "",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Generate the two-chart ICP engager figure.

    Left  — Pie: ICP Match / Non-ICP / Internal / Orbit (count + %)
    Right — Horizontal bar: ICP matches by segment, sorted descending
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator

    counts = {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0}
    segment_counts: dict[str, int] = {}

    for ce in classified:
        cls = ce.classification
        counts[cls] = counts.get(cls, 0) + 1
        if cls == "icp_match" and ce.segment:
            segment_counts[ce.segment] = segment_counts.get(ce.segment, 0) + 1

    total = sum(counts.values())

    fig, (ax_pie, ax_bar) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        facecolor=COLOR_BG,
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.10, wspace=0.35)

    # ── Pie ──────────────────────────────────────────────────────────────────
    pie_labels = ["ICP Match", "Non-ICP", "Internal", "Orbit"]
    pie_values = [counts["icp_match"], counts["non_icp"], counts["internal"], counts["orbit"]]
    pie_colors = [COLOR_ICP, COLOR_NON_ICP, COLOR_INTERNAL, COLOR_ORBIT]

    non_zero = [(v, l, c) for v, l, c in zip(pie_values, pie_labels, pie_colors) if v > 0]
    if non_zero:
        nz_values, nz_labels, nz_colors = zip(*non_zero)
    else:
        nz_values, nz_labels, nz_colors = [1], ["No Data"], ["#DDDDDD"]

    def _autopct(pct: float) -> str:
        count = int(round(pct * total / 100))
        return f"{count}\n({pct:.1f}%)"

    wedges, texts, autotexts = ax_pie.pie(
        nz_values,
        labels=None,
        colors=nz_colors,
        autopct=_autopct,
        startangle=140,
        pctdistance=0.72,
        wedgeprops={"linewidth": 0.8, "edgecolor": COLOR_BG},
    )

    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("#333333")
        at.set_fontfamily("sans-serif")

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(nz_colors, nz_labels)]
    ax_pie.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    ax_pie.set_title(
        "Engager Distribution",
        fontsize=13, fontweight="bold", pad=14, fontfamily="sans-serif", color="#222222",
    )
    ax_pie.set_facecolor(COLOR_BG)

    # ── Horizontal bar ────────────────────────────────────────────────────────
    ax_bar.set_facecolor(COLOR_BG)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    if segment_counts:
        sorted_segments = sorted(segment_counts.items(), key=lambda x: x[1], reverse=True)
        seg_labels = [s[0] for s in sorted_segments]
        seg_values = [s[1] for s in sorted_segments]

        y_pos = range(len(seg_labels))
        bars = ax_bar.barh(y_pos, seg_values, color=COLOR_ICP, height=0.55,
                           edgecolor=COLOR_BG, linewidth=0.4)

        for bar, val in zip(bars, seg_values):
            ax_bar.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=10,
                color="#333333", fontfamily="sans-serif",
            )

        ax_bar.set_yticks(list(y_pos))
        ax_bar.set_yticklabels(seg_labels, fontsize=10, fontfamily="sans-serif", color="#333333")
        ax_bar.invert_yaxis()
        ax_bar.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax_bar.tick_params(axis="x", labelsize=9, color="#CCCCCC")
        ax_bar.tick_params(axis="y", length=0)
        ax_bar.set_xlabel("Engager Count", fontsize=10, color="#666666", fontfamily="sans-serif")
        ax_bar.grid(axis="x", color="#EBEBEB", linewidth=0.6, zorder=0)
        ax_bar.set_axisbelow(True)
        max_val = max(seg_values)
        ax_bar.set_xlim(0, max_val * 1.25 + 0.5)
    else:
        ax_bar.text(
            0.5, 0.5, "No ICP matches\nin this export",
            ha="center", va="center", transform=ax_bar.transAxes,
            fontsize=12, color="#AAAAAA", fontfamily="sans-serif",
        )
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])

    ax_bar.set_title(
        "ICP Matches by Segment",
        fontsize=13, fontweight="bold", pad=14, fontfamily="sans-serif", color="#222222",
    )

    # ── Supra-title ──────────────────────────────────────────────────────────
    subtitle = f"{total} engager{'s' if total != 1 else ''} analysed"
    if title_suffix:
        subtitle += f"  ·  {title_suffix}"
    fig.suptitle(
        f"ICP Engager Report — {company}",
        fontsize=15, fontweight="bold", fontfamily="sans-serif", color="#111111", y=0.97,
    )
    fig.text(0.5, 0.91, subtitle, ha="center", fontsize=10, color="#888888", fontfamily="sans-serif")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_report(fig: "matplotlib.figure.Figure", output_path: str | Path) -> Path:  # type: ignore[name-defined]
    """Save the figure as a PNG and return the resolved path."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=COLOR_BG)
    logger.info("[engager_report] Report saved → %s", out)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 8.  HIGH-LEVEL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(
    company: str,
    csv_path: str | Path,
    output_path: str | Path,
    window_days: int = 0,
    extra_post_ids: Optional[list[str]] = None,
    ordinal_post_id: str = "",
    linkedin_post_url: str = "",
) -> dict:
    """End-to-end pipeline: parse → classify → update history → bridge → chart → save.

    If ``ordinal_post_id`` is provided, the CSV-derived ICP signal is written
    into RuanMei after classification, closing the loop between the analyst's
    engager report and the automated learning system.
    """
    rows = parse_csv(csv_path, window_days=window_days)

    store = load_history(company)
    known = history_names(store)

    classified = classify_engagers(rows, company, known)

    update_history(company, classified, extra_post_ids or [])

    # Bridge into the learning system if a post ID was supplied.
    bridged = False
    if ordinal_post_id:
        bridged = bridge_to_ruan_mei(
            company, classified, ordinal_post_id,
            linkedin_post_url=linkedin_post_url,
        )

    title_suffix = f"(last {window_days}d)" if window_days else ""
    fig = generate_charts(classified, company, title_suffix=title_suffix)
    out = save_report(fig, output_path)

    import matplotlib.pyplot as plt
    plt.close(fig)

    counts: dict[str, int] = {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0}
    segment_counts: dict[str, int] = {}
    for ce in classified:
        counts[ce.classification] = counts.get(ce.classification, 0) + 1
        if ce.classification == "icp_match" and ce.segment:
            segment_counts[ce.segment] = segment_counts.get(ce.segment, 0) + 1

    return {
        "output": str(out),
        "total": len(classified),
        "counts": counts,
        "segments": segment_counts,
        "bridged_to_learning_system": bridged,
        "ordinal_post_id": ordinal_post_id or None,
    }
