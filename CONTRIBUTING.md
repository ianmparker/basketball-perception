# Contributing

Issues and pull requests are welcome. This is a research pipeline, so the bar is
evidence, not polish.

## Ground rules

1. **No footage, frames, or weights.** Do not commit broadcast video, extracted
   frames, trained checkpoints, or anything derived from research-only footage.
   The repository's data policy is in `docs/DATA_POLICY.md`.
2. **No secrets.** Do not commit API keys, bucket credentials, or `.env` files.
3. **Evidence first.** A perception or analytics change should come with a test or a
   scorecard, and the scorecard should state its tolerances and sample size.
4. **Keep the failure list honest.** If a change does not work, add it to
   `docs/KNOWN_FAILURES.md` rather than hiding it.

## Development

```bash
conda env create -f environment.yml
conda activate basketball-perception
python scripts/check_env.py
pytest tests -q
```

## Pull requests

- Keep the diff focused; one problem per PR.
- Include the command you ran and the result.
- Update the README or docs when behavior changes.
- By contributing, you agree that your contribution is licensed under AGPL-3.0.
