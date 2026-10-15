import math

import pytest
import torch
from torch import nn

from benchrep.inspection.gradient_norms import (
    compute_gradient_norm,
    compute_loss_gradient_norm,
    compute_optimizer_gradient_norm,
)


@pytest.mark.parametrize(
    ("norm_type", "expected"),
    [
        ("l1", 18.0),
        ("l2", math.sqrt(86.0)),
        ("inf", 6.0),
    ],
)
def test_norm_combines_all_elements_across_parameter_tensors(
    norm_type, expected,
):
    # Different shapes, negative entries, and an unused parameter.
    gradients = [
        torch.tensor([[3.0, -4.0]], dtype=torch.float64),
        None,
        torch.tensor([5.0, -6.0], dtype=torch.float64),
    ]

    measured = compute_gradient_norm(iter(gradients), norm_type)

    assert measured is not None
    assert measured.item() == pytest.approx(expected)
    assert measured.dtype == torch.float64


@pytest.mark.parametrize(
    ("norm_type", "expected"),
    [("l1", 7.0), ("l2", 5.0), ("inf", 4.0)],
)
def test_sparse_duplicate_indices_are_summed_before_measuring(
    norm_type, expected,
):
    # Entries at index 0 cancel partially: 8 - 5 = 3.
    # The represented dense gradient is therefore [3, -4, 0].
    gradient = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1]]),
        values=torch.tensor([8.0, -5.0, -4.0]),
        size=(3,),
    )

    measured = compute_gradient_norm([gradient], norm_type)

    assert measured is not None
    assert measured.item() == pytest.approx(expected)
    assert not gradient.is_coalesced()
    torch.testing.assert_close(
        gradient._values(), torch.tensor([8.0, -5.0, -4.0]),
    )


@pytest.mark.parametrize("norm_type", ["l1", "l2", "inf"])
def test_missing_gradients_are_distinct_from_zero_or_empty_gradients(norm_type):
    empty_sparse = torch.sparse_coo_tensor(
        indices=torch.empty((1, 0), dtype=torch.long),
        values=torch.empty(0),
        size=(4,),
    )

    assert compute_gradient_norm([None, None], norm_type) is None

    for gradient in (torch.zeros(4), torch.empty(0), empty_sparse):
        measured = compute_gradient_norm([None, gradient], norm_type)
        assert measured is not None
        assert measured.item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_gradients_are_promoted_without_modifying_input(dtype):
    # The L1 norm exceeds float16's maximum finite value.
    gradient = torch.full((4,), 40000.0, dtype=dtype, requires_grad=True)
    original = gradient.detach().clone()
    expected = float(original[0]) * 4

    measured = compute_gradient_norm([gradient], "l1")

    assert measured is not None
    assert measured.dtype == torch.float32
    assert measured.item() == pytest.approx(expected)
    assert not measured.requires_grad
    assert gradient.grad is None
    torch.testing.assert_close(gradient.detach(), original)


@pytest.mark.parametrize("norm_type", ["l1", "l2", "inf"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_gradients_remain_visible(norm_type, nonfinite):
    measured = compute_gradient_norm(
        [torch.tensor([1.0, nonfinite])], norm_type,
    )

    assert measured is not None
    if math.isnan(nonfinite):
        assert torch.isnan(measured).item()
    else:
        assert torch.isinf(measured).item()


@pytest.mark.parametrize("has_accumulated_gradient", [False, True])
def test_loss_inspection_preserves_buffers_and_training_backward(
    has_accumulated_gradient,
):
    parameters = nn.Parameter(torch.tensor([2.0, -3.0]))
    unused = nn.Parameter(torch.tensor([7.0]))
    frozen = nn.Parameter(torch.tensor([11.0]), requires_grad=False)

    if has_accumulated_gradient:
        parameters.grad = torch.tensor([10.0, 20.0])

    original_buffer = parameters.grad
    original_values = (
        original_buffer.clone()
        if original_buffer is not None
        else torch.zeros_like(parameters)
    )

    # Analytic gradient: [2*p0, 3] = [4, 3], whose L2 norm is 5.
    loss = parameters[0].square() + 3 * parameters[1] + frozen.sum()

    # Repeated inspection must also leave the graph usable.
    for _ in range(2):
        result = compute_loss_gradient_norm(
            loss, [parameters, unused, frozen], "l2",
        )

        assert result.skip_reason is None
        assert result.norm is not None
        assert result.norm.item() == pytest.approx(5.0)
        assert not result.norm.requires_grad
        assert parameters.grad is original_buffer
        if original_buffer is not None:
            torch.testing.assert_close(parameters.grad, original_values)
        assert unused.grad is None
        assert frozen.grad is None

    loss.backward()

    torch.testing.assert_close(
        parameters.grad,
        original_values + torch.tensor([4.0, 3.0]),
    )
    assert unused.grad is None
    assert frozen.grad is None


def test_connected_zero_loss_gradient_is_not_reported_as_disconnected():
    connected = nn.Parameter(torch.tensor([2.0]))
    disconnected = nn.Parameter(torch.tensor([3.0]))
    loss = (connected * 0).sum()

    zero_result = compute_loss_gradient_norm(loss, [connected], "l2")
    missing_result = compute_loss_gradient_norm(loss, [disconnected], "l2")

    assert zero_result.skip_reason is None
    assert zero_result.norm is not None
    assert zero_result.norm.item() == 0.0

    assert missing_result.norm is None
    assert "disconnected" in missing_result.skip_reason.lower()


def test_optimizer_inspection_reads_accumulated_gradients_without_changing_them():
    parameters = nn.Parameter(torch.tensor([2.0, -3.0]))
    unused = nn.Parameter(torch.tensor([7.0]))
    frozen = nn.Parameter(torch.tensor([11.0]), requires_grad=False)

    # Two backward passes accumulate [3, 4] + [-1, 2] = [2, 6].
    (parameters * torch.tensor([3.0, 4.0])).sum().backward()
    (parameters * torch.tensor([-1.0, 2.0])).sum().backward()

    # A frozen parameter may retain a stale gradient; exclude it.
    frozen.grad = torch.tensor([1000.0])
    original_buffer = parameters.grad
    original_values = original_buffer.clone()

    result = compute_optimizer_gradient_norm(
        [parameters, unused, frozen], "l2",
    )

    assert result.skip_reason is None
    assert result.norm is not None
    assert result.norm.item() == pytest.approx(math.sqrt(40.0))
    assert parameters.grad is original_buffer
    torch.testing.assert_close(parameters.grad, original_values)
    torch.testing.assert_close(frozen.grad, torch.tensor([1000.0]))
    assert unused.grad is None
