"""Check that the single-notebook implementation cannot silently drift from the package."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _definitions(source, names):
    nodes = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in names:
            nodes.append(ast.dump(node, include_attributes=False))
    return nodes


def test_notebook_implementation_matches_package():
    notebook = json.loads((ROOT / "Causal_AI_Experiment.ipynb").read_text())
    count = 0
    for cell in notebook["cells"]:
        metadata = cell["metadata"]
        if "implementation_source" not in metadata:
            continue
        source = "".join(cell["source"])
        names = metadata["implementation_names"]
        canonical = (ROOT / metadata["implementation_source"]).read_text()
        assert _definitions(source, names) == _definitions(canonical, names)
        assert len(_definitions(source, names)) == len(names)
        count += 1
    assert count == 6


def test_committed_notebook_has_outputs_and_no_local_package_dependency():
    notebook = json.loads((ROOT / "Causal_AI_Experiment.ipynb").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code)
    assert not any(o["output_type"] == "error" for cell in code for o in cell["outputs"])
    assert sum("image/png" in o.get("data", {}) for c in code for o in c["outputs"]) == 5
    for cell in code:
        assert "from causal_hedging" not in "".join(cell["source"])
    counts = [cell["execution_count"] for cell in code]
    assert counts == list(range(1, len(counts) + 1))
