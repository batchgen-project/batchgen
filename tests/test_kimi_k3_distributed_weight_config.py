import ast
from pathlib import Path


WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _class(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(klass, name):
    return next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_distributed_weight_config_reaches_kimi_initializer():
    """The worker and initializer must observe the same distributed store path."""
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    worker = _class(tree, "BatchGenWorker")
    args = _class(tree, "InputArguments")

    fields = {
        node.target.id
        for node in args.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    assert "distributed_weight_config" in fields

    initialize = _method(worker, "_initialize_core_components")
    assignment = next(
        node
        for node in ast.walk(initialize)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "input_arguments"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Dict)

    config_entry = next(
        value
        for key, value in zip(assignment.value.keys, assignment.value.values)
        if isinstance(key, ast.Constant)
        and key.value == "distributed_weight_config"
    )
    assert isinstance(config_entry, ast.Attribute)
    assert isinstance(config_entry.value, ast.Attribute)
    assert isinstance(config_entry.value.value, ast.Name)
    assert config_entry.value.value.id == "self"
    assert config_entry.value.attr == "args"
    assert config_entry.attr == "distributed_weight_config"
