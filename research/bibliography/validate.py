"""Validate the research library and its link to the taxonomy.

Run:
    python research/bibliography/validate.py
    python research/bibliography/validate.py --check-links
    python research/bibliography/validate.py --check-links --strict

The validator is stdlib-only on purpose: it runs in CI and on a fresh machine
without the project environment. The JSON Schema is the contract; this file
adds the rules a schema cannot express:

* keys are unique and stable
* every `track` exists in `research/taxonomy.json`
* an entry that claims `open_access` has a URL
* a paper has at least one of arXiv/DOI/URL
* every `repo` path exists (warning, not error)
* `--check-links` probes HTTP reachability (advisory by default)

Exit codes: 0 OK, 1 validation error, 2 strict link failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RESEARCH_ROOT.parent
DEFAULT_LIBRARY = RESEARCH_ROOT / "bibliography" / "library.json"
DEFAULT_TAXONOMY = RESEARCH_ROOT / "taxonomy.json"
DEFAULT_SCHEMA = RESEARCH_ROOT / "bibliography" / "schema.json"

ACCESS = {"open_access", "preprint", "publisher", "purchase", "subscription", "official_web", "dataset_page"}
TYPES = {"paper", "book", "standard", "dataset", "web"}
RELEVANCE = {"core", "supporting", "background"}
METHODS = {"arxiv-api", "openalex+arxiv-id", "http-head", "citation", "manual"}
ARXIV_RE = re.compile(r"^[0-9]{4}\.[0-9]{4,5}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_library(path: Path | str = DEFAULT_LIBRARY) -> dict:
    return _load(Path(path))


def load_taxonomy(path: Path | str = DEFAULT_TAXONOMY) -> dict:
    return _load(Path(path))


def load_schema(path: Path | str = DEFAULT_SCHEMA) -> dict:
    return _load(Path(path))


def track_ids(taxonomy: dict) -> set[str]:
    return {t["id"] for t in taxonomy["tracks"]}


def _valid_date(value: str) -> bool:
    if not isinstance(value, str) or not DATE_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def validate_entry(entry: dict, tracks: set[str], repo_root: Path = REPO_ROOT) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one entry."""
    errors: list[str] = []
    warnings: list[str] = []
    key = entry.get("key", "<missing>")

    for field in ("key", "type", "title", "authors", "year", "access", "tracks", "relevance", "why", "verified"):
        if field not in entry:
            errors.append(f"{key}: missing required field '{field}'")
    if errors:
        return errors, warnings

    if not KEY_RE.match(entry["key"]):
        errors.append(f"{key}: key must be lowercase kebab-case")
    if entry["type"] not in TYPES:
        errors.append(f"{key}: unknown type '{entry['type']}'")
    if entry["access"] not in ACCESS:
        errors.append(f"{key}: unknown access '{entry['access']}'")
    if entry["relevance"] not in RELEVANCE:
        errors.append(f"{key}: unknown relevance '{entry['relevance']}'")
    if not isinstance(entry["year"], int) or not 1900 <= entry["year"] <= 2100:
        errors.append(f"{key}: year must be an integer in 1900..2100")
    if not entry["authors"] or not all(isinstance(a, str) and a.strip() for a in entry["authors"]):
        errors.append(f"{key}: authors must be a non-empty list of names")
    if len(entry["why"].strip()) < 20:
        errors.append(f"{key}: 'why' must say what this changes for the repo (>=20 chars)")
    if not entry["tracks"]:
        errors.append(f"{key}: at least one track is required")
    unknown = sorted(set(entry["tracks"]) - tracks)
    if unknown:
        errors.append(f"{key}: unknown track(s) {unknown}")

    if entry["access"] in {"open_access", "preprint", "official_web"} and not entry.get("url"):
        errors.append(f"{key}: access={entry['access']} requires a url")
    for field in ("url", "pdf_url"):
        value = entry.get(field)
        if value is not None and not URL_RE.match(str(value)):
            errors.append(f"{key}: {field} must be an http(s) URL")
    if entry.get("arxiv_id") and not ARXIV_RE.match(entry["arxiv_id"]):
        errors.append(f"{key}: arxiv_id must look like 2401.01234")
    if entry["type"] == "paper" and not (entry.get("arxiv_id") or entry.get("doi") or entry.get("url")):
        errors.append(f"{key}: a paper needs an arXiv id, DOI, or URL")

    verified = entry["verified"]
    if not isinstance(verified, dict) or not _valid_date(verified.get("date", "")):
        errors.append(f"{key}: verified.date must be YYYY-MM-DD")
    elif verified.get("method") not in METHODS:
        errors.append(f"{key}: verified.method must be one of {sorted(METHODS)}")

    for rel in entry.get("repo", []):
        target = repo_root / rel.rstrip("/")
        if not target.exists():
            warnings.append(f"{key}: repo path does not exist: {rel}")

    if entry["access"] in {"purchase", "subscription"} and entry.get("pdf_url"):
        warnings.append(f"{key}: paywalled entry should not carry a pdf_url")
    return errors, warnings


def validate_library(library: dict, taxonomy: dict, repo_root: Path = REPO_ROOT) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    entries = library.get("entries")
    if not isinstance(entries, list):
        return ["library.entries must be a list"], warnings, {}

    if not _valid_date(library.get("updated", "")):
        errors.append("library.updated must be YYYY-MM-DD")

    seen: set[str] = set()
    tracks = track_ids(taxonomy)
    counts: dict[str, int] = {}
    for entry in entries:
        e, w = validate_entry(entry, tracks, repo_root)
        errors.extend(e)
        warnings.extend(w)
        key = entry.get("key", "<missing>")
        if key in seen:
            errors.append(f"{key}: duplicate key")
        seen.add(key)
        counts[entry.get("type", "unknown")] = counts.get(entry.get("type", "unknown"), 0) + 1

    return errors, warnings, counts


def check_links(entries: list[dict], timeout: float = 20.0) -> list[tuple[str, str, str]]:
    """Probe every non-purchase URL. Returns (status, key, detail)."""
    results: list[tuple[str, str, str]] = []
    for entry in entries:
        if entry.get("access") in {"purchase", "subscription"}:
            continue
        url = entry.get("url") or entry.get("pdf_url")
        if not url:
            continue
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "basketball-perception-research-librarian/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                results.append(("ok", entry["key"], f"{resp.status} {url}"))
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 405, 429}:
                results.append(("blocked", entry["key"], f"{exc.code} {url}"))
            else:
                results.append(("fail", entry["key"], f"{exc.code} {url}"))
        except Exception as exc:  # noqa: BLE001 - network failures are reported, not raised
            results.append(("fail", entry["key"], f"{type(exc).__name__}: {url}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=str(DEFAULT_LIBRARY))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--check-links", action="store_true", help="probe HTTP(S) links (advisory)")
    parser.add_argument("--strict", action="store_true", help="fail on any non-ok link")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    library = load_library(args.library)
    taxonomy = load_taxonomy(args.taxonomy)
    errors, warnings, counts = validate_library(library, taxonomy)

    link_results: list[tuple[str, str, str]] = []
    if args.check_links:
        link_results = check_links(library["entries"], args.timeout)

    if args.json:
        print(json.dumps({
            "entries": len(library["entries"]),
            "types": counts,
            "tracks": len(taxonomy["tracks"]),
            "errors": errors,
            "warnings": warnings,
            "links": [{"status": s, "key": k, "detail": d} for s, k, d in link_results],
        }, indent=2))
    else:
        if errors:
            print(f"[bibliography] {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  ERROR {err}", file=sys.stderr)
        for warn in warnings:
            print(f"  warn  {warn}", file=sys.stderr)
        for status, key, detail in link_results:
            if status != "ok":
                print(f"  link {status:7} {key}: {detail}", file=sys.stderr)
        if not errors:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"[bibliography] OK: {len(library['entries'])} entries ({summary}), "
                  f"{len(taxonomy['tracks'])} tracks, {len(warnings)} warning(s)")

    if errors:
        return 1
    if args.strict and any(s != "ok" for s, _, _ in link_results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
