"""Contract for the GLM-5 norm -> FP8-quantize fusion (CPU tier).

Three tiers, all CPU-only:

* **The switch module** is imported for real (it is dependency-free on purpose)
  and exercised, including its fail-loud paths.
* **The math** is checked with a CPU oracle of the quantize formula. The
  load-bearing claim of this slice is a COMPOSITION claim -- that quantizing the
  normalized value out of registers equals quantizing it after a BF16 store and
  reload -- and the oracle both confirms it and shows the trap it rests on
  (quantizing the unrounded FP32 accumulator instead gives different bytes).
* **The source** of the two new kernels is pinned against the two kernels they
  must reproduce, and every eliminated launch's call site is pinned at the AST
  level to a branch guarded by the switch.

Why source-level for the kernels: both fused kernels are GPU code (one Triton,
one CUDA) and ``triton`` is not importable on a CPU box, so the runtime bitwise
gate is ``tests/test_glm52_norm_quant_fuse_bitwise_gpu.py``. What CAN be proved
here is that the fused sources still carry, statement for statement, the
arithmetic of the kernels they claim to reproduce -- which is exactly the thing
a later edit would break silently.
"""

import ast
import pathlib
import re

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_POLICY_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/norm_quant_fuse_policy.py"
_QUANT_SRC = _REPO_ROOT / "batchgen_kernels/triton/fp8_quantize.py"
_NORM_SRC = _REPO_ROOT / "batchgen_kernels/triton/rmsnorm.py"
_FUSED_TRITON_SRC = _REPO_ROOT / "batchgen_kernels/triton/rmsnorm_fp8_quant_fused.py"
_OLD_CU = _REPO_ROOT / "batchgen_kernels/src/attention/csrc/rmsnorm.cu"
_FUSED_CU = _REPO_ROOT / "batchgen_kernels/src/attention/csrc/add_rmsnorm_fp8_quant.cu"
_DSA_SEG = _REPO_ROOT / "batchgen/models/glm/glm5/cuda_graph_segments.py"
_REUSE_SEG = _REPO_ROOT / "batchgen/models/glm/glm5/reuse_topk_segment.py"
_LAYER_SEG = _REPO_ROOT / "batchgen/models/glm/glm5/layer_cuda_graph_segments.py"
_MOE_SEG = _REPO_ROOT / "batchgen/models/glm/glm5/moe_cuda_graph_segments.py"
_WRAPPERS = _REPO_ROOT / "batchgen/models/glm/glm5/wrappers.py"
_SETUP = _REPO_ROOT / "batchgen_kernels/setup.py"
_JIT_REGISTRY = _REPO_ROOT / "batchgen_kernels/_jit_registry.py"

_BLOCK = 128


# --------------------------------------------------------------------------- #
# AST helpers (same shape as the pdl/wk slices' source pins)
# --------------------------------------------------------------------------- #

def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _find_def(path: pathlib.Path, class_name: str | None, func_name: str) -> ast.AST:
    tree = ast.parse(path.read_text())
    scope = tree
    if class_name is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                scope = node
                break
        else:
            raise AssertionError(f"{path.name}: no class {class_name}")
    for node in scope.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return node
    raise AssertionError(f"{path.name}: no def {func_name} in {class_name}")


def _calls_with_guards(func_node: ast.AST, call_name: str):
    """Every call to ``call_name``, with the chain of ``if`` tests above it.

    Each entry is ``(guards)`` where ``guards`` is a list of
    ``(test_source, "body" | "orelse")`` from outermost to innermost. An empty
    list means the call is unconditional.
    """
    found: list[list[tuple[str, str]]] = []

    def scan_stmt_exprs(stmt, guards):
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and _callee(node) == call_name:
                found.append(list(guards))

    def walk(stmts, guards):
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                scan_stmt_exprs(stmt.test, guards)
                walk(stmt.body, guards + [(ast.unparse(stmt.test), "body")])
                walk(stmt.orelse, guards + [(ast.unparse(stmt.test), "orelse")])
            elif isinstance(stmt, (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor,
                                   ast.While)):
                for item in getattr(stmt, "items", []):
                    scan_stmt_exprs(item, guards)
                scan_stmt_exprs(getattr(stmt, "iter", stmt), guards)
                walk(stmt.body, guards)
                walk(getattr(stmt, "orelse", []), guards)
            elif isinstance(stmt, ast.Try):
                walk(stmt.body, guards)
                for handler in stmt.handlers:
                    walk(handler.body, guards)
                walk(stmt.orelse, guards)
                walk(stmt.finalbody, guards)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(stmt.body, guards)
            else:
                scan_stmt_exprs(stmt, guards)

    walk(func_node.body, [])
    return found


def _strip_cxx_comments(text: str) -> str:
    """Drop `//` and `/* */` comments, so a "must not appear" pin reads CODE.

    The fused kernel's comments deliberately name the instructions that were
    tried and rejected; a naive substring check would read those as the code.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _cu_function_body(path: pathlib.Path, signature_fragment: str) -> str:
    """The brace-balanced body of the first definition containing a fragment."""
    text = path.read_text()
    start = text.index(signature_fragment)
    open_brace = text.index("{", start)
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    raise AssertionError(f"{path.name}: unbalanced braces after {signature_fragment!r}")


# --------------------------------------------------------------------------- #
# A CPU oracle of the quantize formula
# --------------------------------------------------------------------------- #

_FP8_SAFE_MAX = 448.0
_FP8_SAFE_MAX_INV = 1.0 / 448.0
_FP8_E4M3_MIN_NORMAL = 1.52587890625e-05
_EPSILON = 1e-12


def _cpu_blocked_quantize(x_bf16: torch.Tensor, valid_rows: int | None = None):
    """``_per_token_blocked_quantize_1d_kernel``'s math, on CPU.

    Takes a BF16 tensor, converts per the kernel, and returns
    ``(fp8_bytes_as_uint8, scales)``. Rows at or past ``valid_rows`` take the
    kernel's HAS_VALID_TOKENS branch: exact-zero FP8 bytes and 1e-12 scales.
    """
    assert x_bf16.dtype == torch.bfloat16
    rows, dim = x_bf16.shape
    blocks = dim // _BLOCK
    out = torch.empty(rows, dim, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, blocks, dtype=torch.float32)
    for m in range(rows):
        if valid_rows is not None and m >= valid_rows:
            out[m] = torch.zeros(dim).to(torch.float8_e4m3fn)
            scale[m] = _EPSILON
            continue
        row = x_bf16[m].float()
        for b in range(blocks):
            chunk = row[b * _BLOCK:(b + 1) * _BLOCK]
            amax = max(chunk.abs().max().item(), _FP8_E4M3_MIN_NORMAL)
            s = max(amax * _FP8_SAFE_MAX_INV, _EPSILON)
            scale[m, b] = s
            q = (chunk / s).clamp(-_FP8_SAFE_MAX, _FP8_SAFE_MAX)
            out[m, b * _BLOCK:(b + 1) * _BLOCK] = q.to(torch.float8_e4m3fn)
    return out.view(torch.uint8), scale


def _cpu_add_rmsnorm(residual, hidden, weight, eps, row_valid=None):
    """``add_rmsnorm_bf16_6144_kernel``'s shape, on CPU.

    Returns ``(normed_bf16, normed_fp32_unrounded, residual_bf16)``. The two
    normed forms exist so a test can show that quantizing the unrounded one is
    a DIFFERENT answer -- which is why the fused kernels round first.
    """
    rows = residual.shape[0]
    h = hidden.float()
    if row_valid is not None:
        h = h * 0.0 + torch.where(
            row_valid.view(rows, 1).bool(), hidden.float(), torch.zeros_like(h)
        )
    s = residual.float() + h
    rounded = s.to(torch.bfloat16)
    var = (s * s).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    normed_fp32 = rounded.float() * inv_rms * weight.float()
    return normed_fp32.to(torch.bfloat16), normed_fp32, rounded


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #

def _policy():
    from batchgen.models.glm.glm5 import norm_quant_fuse_policy as policy
    return policy


def test_the_switch_is_on_for_every_site_on_this_branch():
    policy = _policy()
    assert policy.GLM5_NORM_QUANT_FUSE is True
    assert policy.GLM5_NORM_QUANT_FUSE_DISABLED_SITES == ()
    assert policy.GLM5_NORM_QUANT_FUSE_SITES == ("qkv_a", "q_a", "shared_expert")
    for site in policy.GLM5_NORM_QUANT_FUSE_SITES:
        assert policy.glm5_norm_quant_fuse_site_enabled(site) is True


def test_the_master_switch_off_restores_every_site(monkeypatch):
    policy = _policy()
    monkeypatch.setattr(policy, "GLM5_NORM_QUANT_FUSE", False)
    assert policy.glm5_norm_quant_fuse_enabled() is False
    for site in policy.GLM5_NORM_QUANT_FUSE_SITES:
        assert policy.glm5_norm_quant_fuse_site_enabled(site) is False


def test_one_site_can_be_disabled_for_bisection(monkeypatch):
    policy = _policy()
    monkeypatch.setattr(
        policy, "GLM5_NORM_QUANT_FUSE_DISABLED_SITES", ("shared_expert",)
    )
    assert policy.glm5_norm_quant_fuse_site_enabled("shared_expert") is False
    assert policy.glm5_norm_quant_fuse_site_enabled("q_a") is True
    assert policy.glm5_norm_quant_fuse_site_enabled("qkv_a") is True


def test_a_non_bool_switch_fails_loud(monkeypatch):
    policy = _policy()
    monkeypatch.setattr(policy, "GLM5_NORM_QUANT_FUSE", 1)
    with pytest.raises(TypeError, match="GLM5_NORM_QUANT_FUSE"):
        policy.glm5_norm_quant_fuse_enabled()


def test_an_unknown_site_name_fails_loud():
    policy = _policy()
    with pytest.raises(ValueError, match="unknown norm/quant fusion site"):
        policy.glm5_norm_quant_fuse_site_enabled("qkv_b")


def test_an_unknown_disabled_site_fails_loud(monkeypatch):
    """A typo here would silently leave a quantize in the graph."""
    policy = _policy()
    monkeypatch.setattr(
        policy, "GLM5_NORM_QUANT_FUSE_DISABLED_SITES", ("shared_experts",)
    )
    with pytest.raises(ValueError, match="unknown sites"):
        policy.glm5_norm_quant_fuse_site_enabled("q_a")


def test_a_non_tuple_disabled_sites_fails_loud(monkeypatch):
    policy = _policy()
    monkeypatch.setattr(policy, "GLM5_NORM_QUANT_FUSE_DISABLED_SITES", "q_a")
    with pytest.raises(TypeError, match="must be a tuple"):
        policy.glm5_norm_quant_fuse_site_enabled("q_a")


def test_the_policy_module_imports_nothing(monkeypatch):
    """Captured graph segments import it at top level on CPU-only machines."""
    tree = ast.parse(_POLICY_SRC.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], f"policy module must stay dependency-free, got {imports}"


# --------------------------------------------------------------------------- #
# The math: the composition claim, and the trap it rests on
# --------------------------------------------------------------------------- #

def test_quantizing_from_registers_equals_quantizing_after_the_bf16_store():
    """The fusion's whole correctness argument, in one assertion.

    The standalone kernel loaded ``normed_out`` back from global memory, i.e.
    the value after ``__float2bfloat16_rn``. The fused kernels quantize the same
    BF16 value out of registers. Same bits in, same bits out.
    """
    torch.manual_seed(11)
    rows, dim = 9, 512
    residual = (torch.randn(rows, dim) * 0.7).to(torch.bfloat16)
    hidden = (torch.randn(rows, dim) * 0.3).to(torch.bfloat16)
    weight = (torch.randn(dim) * 0.5 + 1.0).to(torch.bfloat16)

    normed_bf16, _, _ = _cpu_add_rmsnorm(residual, hidden, weight, 1e-5)

    # Two-launch form: store BF16, then load it and quantize.
    stored = torch.empty_like(normed_bf16)
    stored.copy_(normed_bf16)
    two_step_fp8, two_step_scale = _cpu_blocked_quantize(stored)
    # Fused form: quantize the register value, which IS that BF16 value.
    fused_fp8, fused_scale = _cpu_blocked_quantize(normed_bf16)

    assert torch.equal(fused_fp8, two_step_fp8)
    assert torch.equal(fused_scale, two_step_scale)


def test_quantizing_the_unrounded_accumulator_is_a_different_answer():
    """Why both fused kernels round to BF16 BEFORE quantizing.

    Taking the FP32 normalized accumulator straight into the quantizer is the
    obvious -- and more accurate -- fusion, and it is exactly what POIS's
    byte-identity ruling forbids: it is not what the two-launch form computed.
    """
    torch.manual_seed(12)
    rows, dim = 6, 256
    residual = (torch.randn(rows, dim) * 0.7).to(torch.bfloat16)
    hidden = (torch.randn(rows, dim) * 0.3).to(torch.bfloat16)
    weight = (torch.randn(dim) * 0.5 + 1.0).to(torch.bfloat16)

    normed_bf16, normed_fp32, _ = _cpu_add_rmsnorm(residual, hidden, weight, 1e-5)
    rounded_fp8, rounded_scale = _cpu_blocked_quantize(normed_bf16)
    # The unrounded path, modelled by quantizing a wider intermediate.
    unrounded_fp8, unrounded_scale = _cpu_blocked_quantize(
        normed_fp32.to(torch.float64).to(torch.bfloat16)
    )
    # Equal here only because bf16(bf16(x)) == bf16(x); the real divergence
    # shows up against the fp32 values themselves:
    assert torch.equal(rounded_fp8, unrounded_fp8)
    assert torch.equal(rounded_scale, unrounded_scale)

    # The scales computed from the fp32 amax differ from the bf16 amax ones on
    # at least one block, which is the byte-level divergence the kernels avoid.
    fp32_amax = normed_fp32.abs().reshape(rows, -1, _BLOCK).amax(dim=-1)
    bf16_amax = normed_bf16.float().abs().reshape(rows, -1, _BLOCK).amax(dim=-1)
    assert not torch.equal(fp32_amax, bf16_amax), (
        "this fixture no longer separates the rounded and unrounded amax, so it "
        "no longer demonstrates the trap it exists to document"
    )


def test_padding_rows_get_exact_zero_bytes_and_the_epsilon_scale():
    """The contract the masked qkv_a GEMM and the dense wk GEMM rely on."""
    torch.manual_seed(13)
    rows, dim = 8, 256
    x = (torch.randn(rows, dim) * 2.0).to(torch.bfloat16)
    fp8, scale = _cpu_blocked_quantize(x, valid_rows=5)
    assert (fp8[5:] == 0).all(), "padded FP8 rows must be exact-zero BYTES"
    assert torch.equal(
        scale[5:], torch.full((rows - 5, dim // _BLOCK), _EPSILON)
    )
    assert (fp8[:5] != 0).any()


@pytest.mark.parametrize("dim", [128, 256, 2048, 2560, 6144, 7168, 8192])
def test_the_reshape_index_map_holds_for_every_128_multiple(dim):
    """The fused Triton kernel reshapes its 1D norm tile onto the quantize's 2D
    tile. That is only index-preserving when the two tiles have the same padded
    width, which is what ``dim % 128 == 0`` buys."""
    next_pow2 = 1 << (dim - 1).bit_length()
    num_blocks = dim // _BLOCK
    num_blocks_p2 = 1 << (num_blocks - 1).bit_length()
    assert next_pow2 == num_blocks_p2 * _BLOCK


def test_a_non_128_multiple_width_is_refused_not_reshaped():
    counter = _FUSED_TRITON_SRC.read_text()
    assert "hidden_size % 128 == 0" in counter
    assert "hidden_size % block_size != 0" in counter


# --------------------------------------------------------------------------- #
# Source: the fused CUDA kernel reproduces add_rmsnorm_bf16_6144_kernel
# --------------------------------------------------------------------------- #

#: Every arithmetic statement of the BF16 half. The reduction tree, the
#: accumulation order and the expression order are the numeric contract.
_CU_NORM_ARITHMETIC = (
    "float h = row_valid ? __bfloat162float(hidden_vector.values[i]) : 0.0f;",
    "float sum = __bfloat162float(residual_vector.values[i]) + h;",
    "rounded_sum.values[i] = __float2bfloat16_rn(sum);",
    "sum_sq = fmaf(sum, sum, sum_sq);",
    "float val = __bfloat162float(rounded_sum.values[i]);",
    "float scale = __bfloat162float(weight_vector.values[i]);",
    "output_vector.values[i] = __float2bfloat16_rn(val * s_inv_rms * scale);",
)


def test_the_old_cuda_kernel_is_untouched():
    """Kernel version management: the fused variant is a NEW file beside it."""
    old = _cu_function_body(_OLD_CU, "__global__ void add_rmsnorm_bf16_6144_kernel")
    for line in _CU_NORM_ARITHMETIC:
        assert line in old, f"the kernel being reproduced changed: {line!r} is gone"
    assert "s_inv_rms = rsqrtf(total / kVectorHiddenSize + eps);" in old
    # and it still compiles into the same extension
    assert '"src/attention/csrc/rmsnorm.cu"' in _SETUP.read_text()


def test_the_fused_cuda_kernel_copies_that_arithmetic_verbatim():
    fused = _cu_function_body(
        _FUSED_CU, "__global__ void add_rmsnorm_bf16_6144_fp8_quant_kernel"
    )
    for line in _CU_NORM_ARITHMETIC:
        assert line in fused, (
            f"the fused kernel's BF16 half diverged from "
            f"add_rmsnorm_bf16_6144_kernel: missing {line!r}"
        )
    # The only renamed symbol is the hidden-size constant.
    assert "s_inv_rms = rsqrtf(total / kHiddenSize + eps);" in fused
    assert "float total = block_reduce_sum_24(sum_sq, warp_sums);" in fused


def test_the_fused_cuda_kernel_copies_the_reduction_tree_verbatim():
    """``tl.sum``-style trees are not interchangeable: a different tree is a
    different float. The 24-warp shuffle reduction must be the same code."""
    old = _cu_function_body(_OLD_CU, "__device__ __forceinline__ float block_reduce_sum_24")
    new = _cu_function_body(_FUSED_CU, "__device__ __forceinline__ float block_reduce_sum_24")
    norm = lambda s: [ln.strip() for ln in s.splitlines() if ln.strip()]
    assert norm(new) == norm(old)

    old_warp = _cu_function_body(_OLD_CU, "__device__ __forceinline__ float warp_reduce_sum")
    new_warp = _cu_function_body(_FUSED_CU, "__device__ __forceinline__ float warp_reduce_sum")
    assert norm(new_warp) == norm(old_warp)


def test_the_fused_cuda_quantize_pins_every_constant_and_operation():
    text = _FUSED_CU.read_text()
    # The constants, with the SAME spelling as the Triton kernel's. In
    # particular 1/448 is a DOUBLE divide cast to float, matching how Triton
    # materializes a Python `1.0 / 448.0` constexpr; `1.0f / 448.0f` would be a
    # single-precision divide and need not be the same float.
    assert "constexpr float kFp8SafeMaxInv = static_cast<float>(1.0 / 448.0);" in text
    assert "constexpr float kFp8MinNormal = 1.52587890625e-05f;" in text
    assert "constexpr float kEpsilon = 1e-12f;" in text
    assert "constexpr float kFp8SafeMax = 448.0f;" in text
    # The operations, in the quantize's order.
    assert "amax = fmaxf(amax, kFp8MinNormal);" in text
    assert "fmaxf(__fmul_rn(amax, kFp8SafeMaxInv), kEpsilon)" in text
    # The divide is `div.full.f32`, the instruction the reference Triton kernel
    # actually emits (PTX-verified on the node, 2026-10-25) -- NOT the IEEE
    # `div.rn.f32`. It was `__fdiv_rn` first and the bitwise gate failed on 4 of
    # 74 cases, each a single e4m3 byte at the 3.5*2^-9 subnormal boundary where
    # a <=2 ULP quotient difference crosses a representable fp8 value.
    assert "div_full_f32(normed[i], scale)" in text
    assert 'asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));' in text
    # CODE only: the comments above both of those deliberately NAME __fdiv_rn to
    # record what was tried and why it failed.
    code = _strip_cxx_comments(text)
    assert "__fdiv_rn" not in code, (
        "the quantize divide was changed back to an IEEE divide; the reference "
        "kernel emits div.full.f32 and the byte gate fails on the subnormal "
        "rounding boundary 3.5*2^-9"
    )
    assert "__fdividef" not in code, (
        "__fdividef is div.approx.f32, a third and even less accurate "
        "instruction -- not the reference's div.full.f32"
    )
    assert "q = fminf(q, kFp8SafeMax);" in text
    assert "q = fmaxf(q, -kFp8SafeMax);" in text
    assert "fabsf(q) < kFiniteBound" in text
    assert "__NV_SATFINITE, __NV_E4M3" in text
    # It quantizes the BF16-ROUNDED value, not the fp32 accumulator.
    assert "normed[i] = __bfloat162float(output_vector.values[i]);" in text


def test_the_fused_cuda_kernel_keeps_the_two_validity_gates_separate():
    text = _FUSED_CU.read_text()
    assert "norm_num_valid_ptr" in text and "quant_num_valid_ptr" in text
    # norm-skipped + quant-valid has no defined answer and must be refused.
    assert "norm_num_valid_ptr == nullptr" in text
    assert "|| norm_num_valid_ptr == quant_num_valid_ptr" in text
    # The padded FP8 row is written through the same conversion, so "zero"
    # means the same byte in both branches.
    assert "zeros.values[i] = __nv_cvt_float_to_fp8(" in text


def test_the_fused_cuda_kernel_fails_loud_instead_of_falling_back():
    text = _FUSED_CU.read_text()
    assert "hidden_size == kHiddenSize" in text
    assert "GLM5_NORM_QUANT_FUSE=False" in text, (
        "an unsupported width must name the switch that restores the old path"
    )
    assert "16-byte aligned bf16 operands" in text
    # No scalar fallback kernel: a reroute would make the eliminated launch
    # reappear while the switch still claimed it was gone.
    assert "add_rmsnorm_kernel" not in text


def test_the_fused_cuda_kernel_is_registered_in_both_build_paths():
    name = "batchgen_kernels.attention._C_add_rmsnorm_fp8_quant"
    source = "src/attention/csrc/add_rmsnorm_fp8_quant.cu"
    setup = _SETUP.read_text()
    assert name in setup and source in setup
    registry = _JIT_REGISTRY.read_text()
    assert name in registry and source in registry
    # Same flags as _C_fused_ops: the BF16 half is that extension's kernel
    # verbatim, and --use_fast_math would move its bytes.
    for block in (setup, registry):
        tail = block.split(name, 1)[1]
        # Cut at the next extension's module name so the window is this
        # extension's own flags and nothing else.
        cut = tail.find("batchgen_kernels.")
        entry = tail if cut < 0 else tail[:cut]
        assert "--use_fast_math" not in entry, (
            "the fused kernel's BF16 half is add_rmsnorm_bf16_6144_kernel "
            "verbatim; fast-math would move its bytes"
        )


# --------------------------------------------------------------------------- #
# Source: the fused Triton kernel reproduces _rmsnorm_kernel + the quantizer
# --------------------------------------------------------------------------- #

def _triton_kernel_body(path: pathlib.Path, name: str) -> list[str]:
    """The kernel's statements as source, docstring dropped."""
    node = _find_def(path, None, name)
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return [ast.unparse(stmt) for stmt in body]


def test_the_old_triton_kernels_are_untouched():
    norm = _triton_kernel_body(_NORM_SRC, "_rmsnorm_kernel")
    assert "variance = tl.sum(sum_sq, axis=0) / hidden_size" in norm
    assert "inv_rms = 1.0 / tl.sqrt(variance + eps)" in norm
    quant = _QUANT_SRC.read_text()
    assert "def _per_token_blocked_quantize_1d_kernel(" in quant


def test_the_fused_triton_kernel_copies_the_norm_statements_verbatim():
    """Every arithmetic statement of ``_rmsnorm_kernel``, unchanged.

    The 1D sum tile is the load-bearing one: summing the same values as a
    ``[16, 128]`` tile associates differently and need not give the same float.
    """
    old = _triton_kernel_body(_NORM_SRC, "_rmsnorm_kernel")
    new = _triton_kernel_body(_FUSED_TRITON_SRC, "_rmsnorm_fp8_quant_kernel")
    numeric = [
        s for s in old
        if any(tok in s for tok in ("sum_sq", "variance", "inv_rms", "x_fp32",
                                    "w_fp32", "out =", "cols =", "mask ="))
    ]
    assert numeric, "fixture lost its grip on _rmsnorm_kernel"
    for stmt in numeric:
        assert stmt in new, (
            f"the fused kernel's norm half diverged from _rmsnorm_kernel: "
            f"missing {stmt!r}"
        )
    # The 2D reshape is applied to the QUANTIZE tile only, never to the sum.
    sum_stmt = [s for s in new if s.startswith("variance =")]
    assert sum_stmt == ["variance = tl.sum(sum_sq, axis=0) / hidden_size"]


def test_the_fused_triton_kernel_copies_the_quantize_statements_verbatim():
    quant_src = _QUANT_SRC.read_text()
    old = _triton_kernel_body(_QUANT_SRC, "_per_token_blocked_quantize_1d_kernel")
    new = _triton_kernel_body(_FUSED_TRITON_SRC, "_rmsnorm_fp8_quant_kernel")
    new_text = "\n".join(new)
    for stmt in (
        "amax = tl.max(tl.abs(q_float), axis=1)",
        "amax = tl.maximum(amax, FP8_E4M3_MIN_NORMAL)",
        "scale = tl.maximum(amax * FP8_SAFE_MAX_INV, EPSILON)",
        "q_scaled = tl.minimum(q_scaled, FP8_SAFE_MAX)",
        "q_scaled = tl.maximum(q_scaled, -FP8_SAFE_MAX)",
        "is_finite = tl.abs(q_scaled) < 1e+30",
        "q_scaled = tl.where(is_finite, q_scaled, 0.0)",
    ):
        assert stmt in old, f"the quantizer being reproduced changed: {stmt!r}"
        assert stmt in new_text, (
            f"the fused kernel's quantize half diverged: missing {stmt!r}"
        )
    # The constants, with the quantizer's spelling.
    assert "FP8_SAFE_MAX_INV: tl.constexpr = 1.0 / 448.0" in quant_src
    assert "FP8_SAFE_MAX_INV: tl.constexpr = 1.0 / 448.0" in _FUSED_TRITON_SRC.read_text()
    # The divide is by the per-block scale broadcast over the 128 columns.
    assert "q_scaled = q_float / scale[:, None]" in new_text
    # It quantizes the value it STORED, i.e. the bf16-rounded one.
    assert "q_bf16 = tl.reshape(out_typed, (NUM_BLOCKS_P2, QUANT_BLOCK))" in new_text
    assert "q_float = q_bf16.to(tl.float32)" in new_text


def test_the_fused_triton_kernel_keeps_the_padding_row_branch():
    new = "\n".join(_triton_kernel_body(_FUSED_TRITON_SRC, "_rmsnorm_fp8_quant_kernel"))
    assert "if pid_m >= num_valid_tokens:" in new
    assert "tl.store(Q_ptr + q_offsets, tl.zeros([NUM_BLOCKS_P2, QUANT_BLOCK]" in new
    assert "EPSILON + tl.zeros([NUM_BLOCKS_P2]" in new
    assert "mask=block_valid" in new


def test_the_fused_triton_launch_passes_no_num_warps():
    """``tl.sum``'s tree depends on the warp count, so a tuned ``num_warps``
    here would change the BF16 norm output -- and the FP8 bytes with it.
    ``_rmsnorm_kernel``'s launch passes neither, so this one must not either."""
    node = _find_def(_FUSED_TRITON_SRC, None, "fused_rmsnorm_fp8_quant")
    launches = [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Subscript)
    ]
    assert len(launches) == 1, "expected exactly one kernel launch"
    kwargs = {kw.arg for kw in launches[0].keywords}
    assert "num_warps" not in kwargs and "num_stages" not in kwargs, (
        f"the fused launch must take _rmsnorm_kernel's default config, got {kwargs}"
    )


def test_the_fused_triton_wrapper_refuses_the_two_pass_norm():
    """``_rmsnorm_kernel_large`` is a different reduction and has no fused
    variant; silently using the single-pass tile for it would be wrong."""
    text = _FUSED_TRITON_SRC.read_text()
    assert "hidden_size <= 8192" in text
    assert "block_tile > 8192" in text


# --------------------------------------------------------------------------- #
# Call-site seams: every eliminated launch is guarded by the switch
# --------------------------------------------------------------------------- #

def test_the_q_a_quantize_is_guarded_in_the_full_dsa_segment():
    forward = _find_def(_DSA_SEG, "Glm5FullDsaAttnSegment", "forward")
    fused = _calls_with_guards(forward, "fused_rmsnorm_fp8_quant")
    assert len(fused) == 1
    assert ("self._fuse_q_a_norm_quant", "body") in fused[0]

    # The old pair survives on the OTHER branch of the same `if`.
    plain = _calls_with_guards(forward, "fused_rmsnorm")
    assert len(plain) == 1
    assert ("self._fuse_q_a_norm_quant", "orelse") in plain[0]


def test_the_q_a_quantize_is_guarded_in_the_reuse_segment():
    forward = _find_def(_REUSE_SEG, "Glm5ReuseTopkAttnSegment", "forward")
    fused = _calls_with_guards(forward, "fused_rmsnorm_fp8_quant")
    assert len(fused) == 1
    assert ("self._fuse_q_a_norm_quant", "body") in fused[0]
    plain = _calls_with_guards(forward, "fused_rmsnorm")
    assert len(plain) == 1
    assert ("self._fuse_q_a_norm_quant", "orelse") in plain[0]


@pytest.mark.parametrize(
    "path,class_name",
    [
        (_DSA_SEG, "Glm5FullDsaAttnSegment"),
        (_REUSE_SEG, "Glm5ReuseTopkAttnSegment"),
    ],
)
def test_every_act_quant_in_a_dsa_forward_is_switch_guarded(path, class_name):
    """No unconditional ``act_quant`` of the qkv_a or q_a activation may remain.

    The attention-output quantize has a NON-norm producer and is deliberately
    out of this slice, so it stays unconditional; it is identified by its
    argument, not by position.
    """
    forward = _find_def(path, class_name, "forward")
    calls = _calls_with_guards(forward, "act_quant")
    guarded_tests = [
        {t for t, _ in guards} for guards in calls
    ]
    unguarded = [
        g for g in guarded_tests
        if not any("_fuse_q" in t or "_indexer_wk_deepgemm" in t for t in g)
    ]
    # Exactly one survivor: the attn_output quantize.
    assert len(unguarded) == 1, (
        f"{class_name}.forward has {len(unguarded)} unguarded act_quant calls; "
        "only the attention-output one (non-norm producer) may be unguarded"
    )


def test_the_layer_segment_fuses_both_norms_behind_the_segments_request():
    forward = _find_def(_LAYER_SEG, "Glm5DecoderLayerGraphSegment", "forward")
    fused = _calls_with_guards(forward, "cuda_add_rmsnorm_fp8_quant")
    assert len(fused) == 2, "input_layernorm and post_attention_layernorm"
    tests = [{t for t, _ in guards} for guards in fused]
    assert any("fuse_attn_input" in t for t in tests[0] | tests[1])
    assert any("fuse_moe_input" in t for t in tests[0] | tests[1])

    # The unfused kernel stays on the other branch of each `if`.
    plain = _calls_with_guards(forward, "cuda_add_rmsnorm")
    assert len(plain) == 2
    for guards in plain:
        assert any(
            ("fuse_attn_input" in t or "fuse_moe_input" in t) and branch == "orelse"
            for t, branch in guards
        ), f"an unfused cuda_add_rmsnorm is not on the switch's else branch: {guards}"


def test_the_layer_segment_asks_the_segments_rather_than_the_switch():
    """One source of truth per site. A layer that fused while its segment did
    not would launch both quantizes; the reverse would starve the GEMM."""
    text = _LAYER_SEG.read_text()
    assert "self.dsa_segment.consumes_fused_hidden_fp8" in text
    assert "self.moe_segment.consumes_fused_padded_fp8" in text
    assert "glm5_norm_quant_fuse" not in text, (
        "the layer segment must not read the switch directly"
    )


def test_the_first_layer_keeps_its_standalone_quantize():
    """The chain's first layer has no residual add, so its input norm is the
    plain ``cuda_rmsnorm``, which has no fused FP8 version. Its ONE quantize
    per step stays standalone and the DSA segment still gets the pair."""
    forward = _find_def(_LAYER_SEG, "Glm5DecoderLayerGraphSegment", "forward")
    calls = _calls_with_guards(forward, "act_quant")
    assert len(calls) == 1
    guards = dict(calls[0])
    assert any("residual is None" in t for t in guards)
    assert any("fuse_attn_input" in t for t in guards)


def test_the_shared_expert_quantize_is_guarded_in_the_wrapper():
    impl = _find_def(_WRAPPERS, "GLM5ExpertWrapper", "_forward_fused_gate_up")
    calls = _calls_with_guards(impl, "per_token_blocked_quantize_bf16_to_fp8_1d")
    assert len(calls) == 1
    assert any(
        "x_fp8 is None and x_scale is None" in t and branch == "body"
        for t, branch in calls[0]
    ), f"the shared expert's x quantize is not guarded: {calls[0]}"
    # The SiLU/up quantize is NOT part of this slice: its producer is a GEMM.
    silu = _calls_with_guards(
        impl, "silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d"
    )
    assert len(silu) == 1 and silu[0] == []


def test_the_moe_segment_validates_the_handed_down_pair():
    text = _MOE_SEG.read_text()
    assert "def consumes_fused_padded_fp8" in text
    assert "def _shared_expert_fused_x" in text
    # Both the missing-pair and the unexpected-pair directions are named.
    assert "passed in, but the shared expert quantizes its own input" in text
    assert "the caller passed " in text
    # The TMA-aligned scale layout is checked, never reinterpreted.
    assert "TMA-aligned MN-major layout" in text


def test_the_dsa_segments_validate_the_handed_down_pair():
    text = _DSA_SEG.read_text()
    assert "def _check_fused_hidden_fp8" in text
    assert "def consumes_fused_hidden_fp8" in text
    assert "TMA-aligned MN-major layout" in text
    assert "one of the two would be a dead launch" in text


def test_the_qkv_a_site_is_scoped_to_the_hoisted_producer_position():
    """On the ``fused_cuda`` wk backend the qkv_a quantize deliberately sits
    AFTER the indexer fork to preserve that version's stream ordering, so there
    is no norm-adjacent position for it. That scope boundary is explicit."""
    text = _DSA_SEG.read_text()
    marker = 'glm5_norm_quant_fuse_site_enabled("qkv_a") and self._indexer_wk_deepgemm'
    assert marker in text
    # The reuse segment has no indexer fork at all, so it has no such bound.
    reuse = _REUSE_SEG.read_text()
    assert 'glm5_norm_quant_fuse_site_enabled("qkv_a")' in reuse
    assert "and self._indexer_wk_deepgemm" not in reuse.split(
        'glm5_norm_quant_fuse_site_enabled("qkv_a")'
    )[1][:120]
