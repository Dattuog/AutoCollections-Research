import json
from copy import deepcopy

from scripts.check_protected import MANIFEST_PATH, ROOT, verify_manifest


def test_protected_manifest_matches_and_detects_tampering() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert verify_manifest(manifest, ROOT) == {
        "missing": [],
        "unexpected": [],
        "changed": [],
    }

    tampered = deepcopy(manifest)
    path = next(iter(tampered["files"]))
    tampered["files"][path] = "wrong"
    assert verify_manifest(tampered, ROOT)["changed"] == [path]
