# Play Taxonomy

`play_taxonomy.json` is the machine-readable vocabulary for play recognition.

- **play_types** — what the offense is running (isolation, PnR, post-up, etc.).
- **coverages** — how the defense is playing it (drop, switch, ice, zone, etc.).
- **play_families** — the formation/action family (horns, Spain PnR, zoom, triangle).

These are **operational heuristics** for tracking and video, not rulebook terms. A
possession can carry multiple labels; `miscellaneous` and broken plays are real classes.

## Rules

- IDs are stable and lowercase kebab-case. Never reuse an ID for a different meaning.
- Add a play type only with a definition, observable signals, and a source/consumer.
- Keep play types, coverages, and families orthogonal. Do not encode "PnR vs drop" as one
  class.
- The taxonomy is a research artifact; the product schema/bindings come later.

## Validate

```powershell
python research/ontology/validate.py
```

## Sources

- `research/knowledge/basketball/06-strategy-and-playbooks.md`
- `research/knowledge/basketball/03-tactics-and-coverage.md`
- NBA/Synergy play-type vocabulary (see `synergy-sportradar` in the library)
