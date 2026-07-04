# CUTLASS 4.0 CollectiveBuilder Instantiations for GEMM Microbenchmarks

**CUTLASS Commit**: `b995f933179c22d3fe0d871c3a53d11e4681950f` (v4.0.0)

---

## PART A: FP8 (e4m3) DENSE GEMM — All Architectures

### SM90 (Hopper) — FP8 e4m3 Dense GEMM

**Source**: [54_hopper_fp8_warp_specialized_gemm.cu](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/examples/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu#L98-L165)

```cpp
// A matrix configuration
using ElementA           = cutlass::float_e4m3_t;
using LayoutA            = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // = 16

// B matrix configuration
using ElementB           = cutlass::float_e4m3_t;
using LayoutB            = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // = 16

// C/D matrix configuration
using ElementC           = cutlass::float_e4m3_t;
using LayoutC            = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // = 16

using ElementD           = ElementC;
using LayoutD            = LayoutC;
constexpr int AlignmentD = AlignmentC;

// Core kernel configurations
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;
using TileShape          = Shape<_128, _128, _128>;
using ClusterShape       = Shape<_1, _2, _1>;
using KernelSchedule     = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

**Key Points**:
- **OpClass**: `OpClassTensorOp` (standard tensor core)
- **Alignment**: 16 elements (128 bits / 8 bits per e4m3)
- **TileShape**: 128×128×128 (M×N×K)
- **ClusterShape**: 1×2×1 (1 SM in M, 2 SMs in N, 1 in K)
- **KernelSchedule**: `KernelTmaWarpSpecializedCooperative`
- **No scale factors** (dense FP8, not blockscaled)

---

### SM120 (Blackwell GeForce) — FP8 e4m3 Dense GEMM

**Status**: ❌ **NOT SUPPORTED** — SM120 has no native f16 dense collective and no native FP8 dense collective. Only blockscaled (NVFP4) is supported.

**Workaround**: Use NVFP4 blockscaled (see Part B below).

---

### SM100 (Blackwell Data Center) — FP8 e4m3 Dense GEMM

**Status**: ❌ **NOT SUPPORTED** — SM100 has no native FP8 dense collective. Only blockscaled (NVFP4) is supported.

**Workaround**: Use NVFP4 blockscaled (see Part B below).

---

## PART B: NVFP4 BLOCKSCALED GEMM — SM120 & SM100

### SM120 (Blackwell GeForce) — NVFP4 Blockscaled GEMM

**Source**: [79a_blackwell_geforce_nvfp4_bf16_gemm.cu](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu#L99-L150)

```cpp
// A matrix configuration (NVFP4 blockscaled)
using ElementA           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag         = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;  // 32 elements = 128 bits (4-bit data)

// B matrix configuration (NVFP4 blockscaled)
using ElementB           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag         = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// C/D matrix configuration (output in BF16)
using ElementC           = cutlass::bfloat16_t;
using LayoutCTag         = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // = 8

using ElementD           = cutlass::bfloat16_t;
using LayoutDTag         = cutlass::layout::RowMajor;
constexpr int AlignmentD = 8;

// Core kernel configurations
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape   = Shape<_128, _128, _128>;  // M×N×K
using ClusterShape       = Shape<_1, _1, _1>;        // 1×1×1 (no multicast on GeForce)

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

**Scale Factor Types & Layout**:

From [sm120_blockscaled_mma_builder.inl](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl#L83-L225):

```cpp
using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
// For NVFP4: ElementSFA = ElementSFB = cutlass::float_e2m1_t (2-bit scale factor)

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
// These are interleaved layouts (NOT strides) — extracted from builder
```

**Arguments Construction** ([line 377-391](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu#L377-L391)):

```cpp
typename Gemm::Arguments arguments {
  cutlass::gemm::GemmUniversalMode::kGemm,
  {options.m, options.n, options.k, 1},
  { // Mainloop arguments
    block_A.device_data(), stride_A,
    block_B.device_data(), stride_B,
    block_SFA.device_data(), layout_SFA,    // Scale factor for A
    block_SFB.device_data(), layout_SFB     // Scale factor for B
  },
  { // Epilogue arguments
    {options.alpha, options.beta},
    block_C.device_data(), stride_C,
    block_D.device_data(), stride_D
  }
};
```

**Data Allocation** ([line 183-186](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu#L183-L186)):

```cpp
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
```

**Key Points**:
- **OpClass**: `OpClassBlockScaledTensorOp` (block-scaled tensor core)
- **ElementA/B**: `cutlass::nv_float4_t<cutlass::float_e2m1_t>` (4-bit data + 2-bit scale)
- **Alignment**: 32 elements (128 bits / 4 bits per element)
- **ThreadBlockShape**: 128×128×128
- **ClusterShape**: 1×1×1 (GeForce RTX 50 does NOT support multicast)
- **Scale Factors**: Passed as separate tensors in mainloop args
  - `block_SFA.device_data()` → pointer to scale factor tensor for A
  - `block_SFB.device_data()` → pointer to scale factor tensor for B
  - `layout_SFA`, `layout_SFB` → interleaved layouts (NOT strides)

---

### SM100 (Blackwell Data Center) — NVFP4 Blockscaled GEMM

**Source**: [72a_blackwell_nvfp4_bf16_gemm.cu](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu#L96-L147)

```cpp
// A matrix configuration (NVFP4 blockscaled)
using ElementA           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag         = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

// B matrix configuration (NVFP4 blockscaled)
using ElementB           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag         = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// C/D matrix configuration (output in BF16)
using ElementC           = cutlass::bfloat16_t;
using LayoutCTag         = cutlass::layout::RowMajor;
constexpr int AlignmentC = 8;

using ElementD           = cutlass::bfloat16_t;
using LayoutDTag         = cutlass::layout::RowMajor;
constexpr int AlignmentD = 8;

// Core kernel configurations
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape       = Shape<_256, _256, _256>;  // M×N×K (larger tile for SM100)
using ClusterShape       = Shape<_4, _4, _1>;        // 4×4×1 (2SM per cluster in M/N)

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

**Scale Factor Types & Layout**:

From [sm100_blockscaled_umma_builder.inl](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L133-L234):

```cpp
using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
// For NVFP4: ElementSFA = ElementSFB = cutlass::float_e2m1_t

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
// Interleaved layouts (NOT strides)
```

**Arguments Construction** ([line 378-391](https://github.com/NVIDIA/cutlass/blob/b995f933179c22d3fe0d871c3a53d11e4681950f/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu#L378-L391)):

```cpp
typename Gemm::Arguments arguments {
  cutlass::gemm::GemmUniversalMode::kGemm,
  {options.m, options.n, options.k, 1},
  { // Mainloop arguments
    block_A.device_data(), stride_A,
    block_B.device_data(), stride_B,
    block_SFA.device_data(), layout_SFA,
    block_SFB.device_data(), layout_SFB
  },
  { // Epilogue arguments
    {options.alpha, options.beta},
    block_C.device_data(), stride_C,
    block_D.device_data(), stride_D
  }
};
```

**Key Points**:
- **OpClass**: `OpClassBlockScaledTensorOp`
- **ElementA/B**: `cutlass::nv_float4_t<cutlass::float_e2m1_t>`
- **Alignment**: 32 elements
- **MmaTileShape**: 256×256×256 (larger than SM120 due to TMEM)
- **ClusterShape**: 4×4×1 (2 SMs per cluster in M/N, 1 in K)
  - **Note**: This is "2SM" mode — each cluster spans 2 SMs in M and N
  - Yields **tcgen05 + TMEM** (Tensor Memory) for efficient data staging
- **Scale Factors**: Same structure as SM120 (separate tensors with interleaved layouts)

---

## Summary Table

| Arch | Dtype | OpClass | ElementA/B | Alignment | TileShape | ClusterShape | Notes |
|------|-------|---------|-----------|-----------|-----------|--------------|-------|
| SM90 | FP8 e4m3 | OpClassTensorOp | `float_e4m3_t` | 16 | 128×128×128 | 1×2×1 | Dense GEMM, no scale factors |
| SM120 | NVFP4 | OpClassBlockScaledTensorOp | `nv_float4_t<float_e2m1_t>` | 32 | 128×128×128 | 1×1×1 | Blockscaled, scale factors in args |
| SM100 | NVFP4 | OpClassBlockScaledTensorOp | `nv_float4_t<float_e2m1_t>` | 32 | 256×256×256 | 4×4×1 | Blockscaled, 2SM mode, TMEM |

---

## Scale Factor Plumbing (NVFP4 Only)

**Mainloop Arguments Structure**:
```cpp
struct MainloopArguments {
  ElementA* ptr_A;
  StrideA stride_A;
  ElementB* ptr_B;
  StrideB stride_B;
  ElementSFA* ptr_SFA;      // Scale factor tensor for A
  LayoutSFA layout_SFA;     // Interleaved layout (NOT stride)
  ElementSFB* ptr_SFB;      // Scale factor tensor for B
  LayoutSFB layout_SFB;     // Interleaved layout (NOT stride)
};
```

**Key Insight**: Scale factors are **NOT** passed as strides but as **full layout objects** extracted from the builder. This allows the kernel to handle the complex interleaved block-scaled layout automatically.

**Allocation Pattern**:
```cpp
// Data tensors (4-bit packed)
cutlass::HostTensor<ElementA::DataType, PackedVectorLayout> block_A;
cutlass::HostTensor<ElementB::DataType, PackedVectorLayout> block_B;

// Scale factor tensors (2-bit, separate from data)
cutlass::HostTensor<ElementA::ScaleFactorType, PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::ScaleFactorType, PackedVectorLayout> block_SFB;
```

---

## Compilation Flags

```bash
# SM90 (Hopper)
-gencode arch=compute_90,code=sm_90

# SM120 (Blackwell GeForce)
-gencode arch=compute_120,code=sm_120

# SM100 (Blackwell Data Center)
-gencode arch=compute_100,code=sm_100
```

