"""Query the research library and knowledge index.

Examples:
    python research/query.py stats
    python research/query.py search occlusion
    python research/query.py track perception.calibration
    python research/query.py code tracking/
    python research/query.py gaps
    python research/query.py cite arxiv-2110.06864
    python research/query.py note perception.calibration.homography
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent
LIBRARY = RESEARCH_ROOT / "bibliography" / "library.json"
INDEX = RESEARCH_ROOT / "knowledge" / "index.json"
TAXONOMY = RESEARCH_ROOT / "taxonomy.json"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the relevant build script first")
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _overlap(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return bool(a and b and (a in b or b in a))


def _authors(entry: dict, limit: int = 3) -> str:
    names = entry["authors"]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + ", et al."


def _print_entries(entries: list[dict]) -> None:
    if not entries:
        print("  (none)")
        return
    for e in entries:
        flags = f"{e['relevance']:<10} {e['access']:<12}"
        print(f"  {e['key']:<24} {e['year']}  {flags} {e['title'][:88]}")
        print(f"    {e['why']}")
        if e.get("url"):
            print(f"    {e['url']}")


def cmd_stats(_: argparse.Namespace) -> int:
    lib = _load(LIBRARY)
    idx = _load(INDEX)
    tax = _load(TAXONOMY)
    entries = lib["entries"]
    print(f"Library: {len(entries)} entries, {len(tax['tracks'])} tracks")
    by_type: dict[str, int] = {}
    by_access: dict[str, int] = {}
    by_rel: dict[str, int] = {}
    for e in entries:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        by_access[e["access"]] = by_access.get(e["access"], 0) + 1
        by_rel[e["relevance"]] = by_rel.get(e["relevance"], 0) + 1
    print("  types    " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print("  access   " + ", ".join(f"{k}={v}" for k, v in sorted(by_access.items())))
    print("  relevance " + ", ".join(f"{k}={v}" for k, v in sorted(by_rel.items())))
    print(f"Knowledge: {idx['stats']['notes']} notes, {idx['stats']['words']:,} words")
    print("  " + ", ".join(f"{k}={v}" for k, v in idx["stats"]["by_status"].items()))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    lib = _load(LIBRARY)
    needle = _norm(args.query)
    hits = []
    for e in lib["entries"]:
        hay = " ".join([
            e["key"], e["title"], " ".join(e["authors"]), str(e.get("venue") or ""),
            e["why"], " ".join(e.get("topics", [])), " ".join(e["tracks"]),
        ])
        if needle in _norm(hay):
            hits.append(e)
    hits.sort(key=lambda e: ({"core": 0, "supporting": 1, "background": 2}[e["relevance"]], e["year"]))
    print(f"{len(hits)} match(es) for {args.query!r}")
    _print_entries(hits[: args.limit])
    if len(hits) > args.limit:
        print(f"  ... {len(hits) - args.limit} more (raise --limit)")
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    lib = _load(LIBRARY)
    idx = _load(INDEX)
    tax = _load(TAXONOMY)
    track = next((t for t in tax["tracks"] if t["id"] == args.track), None)
    if not track:
        print(f"unknown track {args.track!r}; try: {', '.join(t['id'] for t in tax['tracks'])}", file=sys.stderr)
        return 2
    print(f"{track['name']}  [{track['id']}]")
    print(f"  {track['question']}")
    entries = [e for e in lib["entries"] if args.track in e["tracks"]]
    notes = [n for n in idx["notes"] if n["track"] == args.track]
    print(f"\nReading ({len(entries)}):")
    _print_entries(entries)
    print(f"\nNotes ({len(notes)}):")
    for n in notes:
        print(f"  {n['id']:<44} {n['status']:<7} {n['path']}")
    if not notes:
        print("  (no note yet — this is a gap)")
    return 0


def cmd_code(args: argparse.Namespace) -> int:
    lib = _load(LIBRARY)
    idx = _load(INDEX)
    target = args.path.rstrip("/")
    print(f"Research attached to {target}:")
    entries = [e for e in lib["entries"] if any(_overlap(r, target) for r in e.get("repo", []))]
    _print_entries(entries)
    notes = [n for n in idx["notes"] if any(_overlap(c, target) for c in n.get("code", []))]
    print(f"\nNotes ({len(notes)}):")
    for n in notes:
        print(f"  {n['id']:<44} {n['path']}")
    return 0


def cmd_gaps(_: argparse.Namespace) -> int:
    lib = _load(LIBRARY)
    idx = _load(INDEX)
    tax = _load(TAXONOMY)
    print(f"{'track':<34} {'core':>4} {'supp':>5} {'bg':>3} {'notes':>6}  status")
    gaps = 0
    for t in tax["tracks"]:
        entries = [e for e in lib["entries"] if t["id"] in e["tracks"]]
        core = sum(e["relevance"] == "core" for e in entries)
        supp = sum(e["relevance"] == "supporting" for e in entries)
        bg = sum(e["relevance"] == "background" for e in entries)
        notes = sum(n["track"] == t["id"] for n in idx["notes"])
        missing = []
        if core == 0:
            missing.append("no core paper")
        if notes == 0:
            missing.append("no note")
        if missing:
            gaps += 1
        print(f"{t['id']:<34} {core:>4} {supp:>5} {bg:>3} {notes:>6}  {', '.join(missing) or 'ok'}")
    print(f"\n{gaps} track(s) need attention.")
    return 0


def cmd_cite(args: argparse.Namespace) -> int:
    lib = _load(LIBRARY)
    entry = next((e for e in lib["entries"] if e["key"] == args.key), None)
    if not entry:
        print(f"unknown key {args.key!r}", file=sys.stderr)
        return 2
    venue = entry.get("venue") or entry["type"]
    bits = [f"{_authors(entry)} ({entry['year']}).", f"{entry['title']}.", f"{venue}."]
    if entry.get("arxiv_id"):
        bits.append(f"arXiv:{entry['arxiv_id']}.")
    if entry.get("doi"):
        bits.append(f"https://doi.org/{entry['doi']}")
    elif entry.get("url"):
        bits.append(entry["url"])
    print(" ".join(bits))
    print(f"  access: {entry['access']}  license: {entry.get('license', 'unknown')}")
    return 0


def cmd_note(args: argparse.Namespace) -> int:
    idx = _load(INDEX)
    note = next((n for n in idx["notes"] if n["id"] == args.id), None)
    if not note:
        print(f"unknown note {args.id!r}", file=sys.stderr)
        return 2
    print(f"{note['title']}  [{note['id']}]")
    print(f"  {note['path']}  ({note['status']}, {note['confidence']} confidence, {note['words']} words)")
    print(f"  sources: {', '.join(note['sources']) or '(none)'}")
    print(f"  code:    {', '.join(note['code']) or '(none)'}")
    print(f"  heads:   {'; '.join(note['headings'][:8])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="library and knowledge counts")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="full-text search over titles, authors, why, topics, tracks")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("track", help="everything attached to one taxonomy track")
    p.add_argument("track")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("code", help="research attached to a repo path")
    p.add_argument("path")
    p.set_defaults(func=cmd_code)

    p = sub.add_parser("gaps", help="tracks without a core paper or a note")
    p.set_defaults(func=cmd_gaps)

    p = sub.add_parser("cite", help="render one citation")
    p.add_argument("key")
    p.set_defaults(func=cmd_cite)

    p = sub.add_parser("note", help="show one knowledge note")
    p.add_argument("id")
    p.set_defaults(func=cmd_note)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
