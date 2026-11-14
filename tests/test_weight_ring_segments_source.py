"""CPU-only source checks for the weight ring's slot allocation.

Every slot of a module type's ring is carved from one allocation per grow,
with the type's tensors at fixed, aligned offsets ordered by name. That
layout is identical on every rank, which a node-local replica transport
needs in order to address a peer's slot, and a grow never moves an
existing slot, which CUDA graphs that captured slot pointers need.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUFFER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp"


def _function(source: str, signature: str) -> str:
    # The live definition starts a line; commented-out copies start "//".
    start = source.index("\n" + signature) + 1
    depth = 0
    for index in range(source.index("{", start), len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated {signature}")


def test_every_ring_allocation_goes_through_one_helper():
    source = BUFFER.read_text()
    for signature in (
        "void GPU_Weight_Buffer::Init()",
        "void GPU_Weight_Buffer::resize_buffer()",
        "void GPU_Weight_Buffer::reset_prefill_buffer()",
        "void GPU_Weight_Buffer::reset_decoding_buffer()",
    ):
        body = _function(source, signature)
        assert "append_ring_slots_(" in body, signature
        assert "torch::zeros" not in body, signature


def test_helper_carves_one_allocation_with_ordered_aligned_fields():
    body = _function(
        BUFFER.read_text(), "void GPU_Weight_Buffer::append_ring_slots_("
    )
    assert body.count("torch::zeros(") == 1
    assert "std::sort(names.begin(), names.end());" in body
    assert "constexpr int64_t kAlign = 256;" in body
    # Field-major arena: each field owns one contiguous block across all
    # slots (a frame's slots then expose each field as a dense operand), and
    # a slot's field is addressed at block base + slot * per-field stride.
    assert "(nbytes + kAlign - 1) / kAlign * kAlign;" in body
    assert "total += count * field_stride;" in body
    narrow = body.index(".narrow(0,")
    assert body.index("field.offset +", narrow) > narrow
    assert body.index("index * field_stride", narrow) > narrow
    assert body.index(".view(field.dtype)") > narrow
    # A grow appends; it never clears or reallocates existing slots.
    assert "buffers.push_back(" in body
    assert "buffers.clear()" not in body


def test_grow_appends_without_touching_existing_slots():
    body = _function(BUFFER.read_text(), "void GPU_Weight_Buffer::resize_buffer()")
    append = body.index('this->append_ring_slots_("routed_expert", to_add_buffer)')
    grow = body.index('this->slots_.grow_type(')
    assert append < grow
    assert ".clear()" not in body
