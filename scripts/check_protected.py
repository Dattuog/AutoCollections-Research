import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocollections.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/protected_manifest.json"
PROTECTED_FILES = (
    "AUTOCOLLECTIONS_CODEX_IMPLEMETATION.md",
    "program.md",
    "prepare.py",
    "pyproject.toml",
    "uv.lock",
    "configs/benchmark.yaml",
    "configs/business_simulation.yaml",
    "configs/business_simulation_v2.yaml",
    "reports/baselines.json",
    "reports/policy_baselines.json",
    "reports/simulator_sensitivity.json",
    "reports/policy_baselines_v2.json",
    "reports/simulator_sensitivity_v2.json",
    "reports/feasibility_audit_v2.json",
    "reports/evaluator_identity.json",
    "reports/final_selection_manifest.json",
)
PROTECTED_DIRECTORIES = ("autocollections", "scripts", "tests")


def protected_hashes(root: Path = ROOT) -> dict[str, str]:
    paths = [root / name for name in PROTECTED_FILES]
    for directory in PROTECTED_DIRECTORIES:
        paths.extend((root / directory).rglob("*.py"))
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Protected files are missing: {missing}")
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(set(paths))}


def verify_manifest(manifest: dict, root: Path = ROOT) -> dict[str, list[str]]:
    expected = manifest["files"]
    actual = protected_hashes(root)
    return {
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "changed": sorted(
            path for path in set(expected) & set(actual) if expected[path] != actual[path]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify protected AutoCollections files")
    parser.add_argument("--write", action="store_true", help="write the initial manifest")
    args = parser.parse_args()
    if args.write:
        manifest = {"manifest_version": 1, "files": protected_hashes()}
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)} with {len(manifest['files'])} files")
        return

    problems = verify_manifest(json.loads(MANIFEST_PATH.read_text()))
    if any(problems.values()):
        raise SystemExit(f"PROTECTED INTEGRITY FAIL: {problems}")
    print("PROTECTED INTEGRITY PASS")


if __name__ == "__main__":
    main()
