"""Check that the project environment has the runtime the pipeline needs.

Run:
    python scripts/check_env.py

Exit codes: 0 OK, 1 a required package is missing.
"""
from __future__ import annotations

import importlib
import sys


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    failures = 0

    def check(label: str, module: str, required: bool = True) -> object | None:
        nonlocal failures
        try:
            mod = importlib.import_module(module)
            print(f"[OK]   {label}")
            return mod
        except Exception as exc:  # noqa: BLE001
            level = "FAIL" if required else "warn"
            print(f"[{level}] {label}: {type(exc).__name__}: {exc}")
            if required:
                failures += 1
            return None

    print(f"Python {sys.version.split()[0]}  ({sys.executable})")

    torch = check("torch", "torch")
    if torch is not None:
        cuda = bool(torch.cuda.is_available())
        print(f"       torch {torch.__version__}  cuda={torch.version.cuda}  available={cuda}")
        if not cuda:
            print("[warn] torch is CPU-only or CUDA is unavailable; the pipeline will be slow")
        else:
            try:
                print(f"       GPU: {torch.cuda.get_device_name(0)}")
            except Exception:  # noqa: BLE001
                pass

    cv2 = check("OpenCV", "cv2")
    if cv2 is not None:
        print(f"       cv2 {cv2.__version__}")

    ultra = check("Ultralytics", "ultralytics")
    if ultra is not None:
        print(f"       ultralytics {ultra.__version__}")

    pyd = check("pydantic", "pydantic")
    if pyd is not None:
        print(f"       pydantic {pyd.__version__}")

    check("numpy", "numpy")
    check("pandas", "pandas")
    check("boxmot", "boxmot", required=False)
    check("open_clip", "open_clip", required=False)
    check("dvc", "dvc", required=False)

    print()
    if failures:
        print(f"[check_env] {failures} required check(s) failed")
        return 1
    print("[check_env] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
