"""Validate the source registry and enforce the governance gates.

Run:
    python registry/validate.py

The JSON Schema is the machine-readable contract; these pydantic models are the
runtime validator used by tests and CI. They intentionally check a few rules
the schema cannot express:

* a commercially-usable source (SAFE/CAUTION) needs `approved_by` and `approved_date`
* a DO-NOT-USE source cannot be approved
* a removed source must not be training-eligible

License fields are optional metadata. They are recorded when known, but they do
not gate intake: this project is research-only and nothing here is released.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "registry" / "sources.json"


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str
    source_url: str
    source_platform: str
    contents: str = ""
    size: str = ""
    license_id: str = "unrecorded"
    license_url: Optional[str] = None
    provenance_confidence: Literal[
        "verified", "self-asserted", "unknown", "likely-infringing"
    ] = "unknown"
    commercial_use_allowed: Optional[bool] = None
    modification_allowed: Optional[bool] = None
    redistribution_allowed: Optional[bool] = None
    ml_training_allowed: Optional[bool] = None
    attribution_text: str = ""
    retrieval_method: str = ""
    retrieval_date: Optional[str] = None
    risk_tier: Literal["SAFE", "CAUTION", "RESEARCH-ONLY", "DO-NOT-USE"] = "RESEARCH-ONLY"
    relevance: Literal["high", "medium", "low"] = "medium"
    notes: str = ""
    status: Literal["staged", "under-review", "approved", "rejected", "removed"]
    approved_by: Optional[str] = None
    approved_date: Optional[str] = None
    takedown_status: Literal["active", "pending", "removed"] = "active"

    @model_validator(mode="after")
    def _governance_gates(self) -> "SourceRecord":
        if self.status == "approved":
            if self.risk_tier in ("SAFE", "CAUTION"):
                if not self.approved_by or not self.approved_date:
                    raise ValueError(
                        f"{self.asset_id}: commercially-usable sources need "
                        "approved_by and approved_date"
                    )
            if self.risk_tier == "DO-NOT-USE":
                raise ValueError(f"{self.asset_id}: DO-NOT-USE sources cannot be approved")
        if self.status == "removed" and self.takedown_status == "active":
            raise ValueError(
                f"{self.asset_id}: removed sources must have takedown_status != active"
            )
        return self


class SourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    sources: List[SourceRecord]


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> SourceRegistry:
    with open(path, encoding="utf-8") as f:
        return SourceRegistry.model_validate(json.load(f))


def main() -> int:
    registry = load_registry()
    ids = [s.asset_id for s in registry.sources]
    duplicates = sorted({x for x in ids if ids.count(x) > 1})
    if duplicates:
        print(f"[registry] duplicate asset_id(s): {duplicates}", file=sys.stderr)
        return 1
    tiers = {
        tier: sum(1 for s in registry.sources if s.risk_tier == tier)
        for tier in ("SAFE", "CAUTION", "RESEARCH-ONLY", "DO-NOT-USE")
    }
    print(f"[registry] OK: {len(registry.sources)} sources, tiers={tiers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
