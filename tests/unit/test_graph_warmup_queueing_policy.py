"""The decode-configure graph-warmup predicate on a queueing batch.

A batch larger than the decode capacity reaches the predicate with
``global_batch_has_queueing=True`` and no graph manager yet; that branch must
answer from the CLI graph request, for every model family.
"""

import inspect

import pytest

from batchgen.models.glm.glm5.cuda_graph_policy import (
    should_warmup_cuda_graphs_before_decode,
)


@pytest.mark.parametrize(
    "model_name,enable,expected",
    [
        ("zai-org/GLM-5-FP8", True, True),
        ("zai-org/GLM-5-FP8", False, False),
        ("moonshotai/Kimi-K3.1-Pro", True, False),
        ("gpt-oss-120b", True, False),
        (None, True, False),
    ],
)
def test_queueing_batch_without_graph_manager_answers_from_cli(
    model_name, enable, expected
):
    assert (
        should_warmup_cuda_graphs_before_decode(
            graph_manager_is_initialized=False,
            global_batch_has_queueing=True,
            model_name=model_name,
            enable_cuda_graph=enable,
        )
        is expected
    )


def test_initialized_graph_manager_never_warms_up_again():
    assert not should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=True,
        global_batch_has_queueing=True,
        model_name="zai-org/GLM-5-FP8",
        enable_cuda_graph=True,
    )


def test_non_queueing_batch_without_graph_manager_warms_up():
    assert should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=False,
        global_batch_has_queueing=False,
        model_name="gpt-oss-120b",
    )


def test_predicate_takes_no_environment_mapping():
    assert list(inspect.signature(should_warmup_cuda_graphs_before_decode).parameters) == [
        "graph_manager_is_initialized",
        "global_batch_has_queueing",
        "model_name",
        "enable_cuda_graph",
    ]
