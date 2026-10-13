# Third-party runtime licenses

This note records the licenses that apply to the GLM-5.3 FA3 runtime
dependencies assembled by the deployment guide. It supplements BatchGen's
Apache 2.0 license; it does not change the license of BatchGen source code.

| Component | Version/source | License and required attribution |
| --- | --- | --- |
| SGL DeepGEMM packaging and fork changes | `sgl-deep-gemm` 0.1.5.post3, SGLang DeepGEMM fork | Apache License 2.0; retain the fork's `LICENSE` and any modification notices. |
| DeepGEMM source copied into the package | DeepSeek DeepGEMM sources used by the fork | MIT License, copyright DeepSeek (2025); retain the MIT copyright and permission notice. |
| CUTLASS and CuTe headers bundled for JIT/runtime use | CUTLASS submodule pinned by the fork | BSD 3-Clause, copyright NVIDIA Corporation & affiliates (2017–2025); retain the notice and disclaimer. The CUTLASS Python CuTeDSL directory has separate NVIDIA EULA terms and is not used by this BatchGen runtime. |
| `{fmt}` headers | `{fmt}` submodule pinned by the fork | MIT License, copyright Victor Zverovich and `{fmt}` contributors; retain the notice and the stated object-code exception. |

When distributing a built `sgl-deep-gemm` wheel or a container that includes
its bundled headers, ship readable copies of the four source license texts
above (or preserve the equivalent license files in the dependency bundle).
Do not describe this dependency as SGLang itself: BatchGen uses the
`deep_gemm` package and its CUDA extension, not the SGLang serving runtime.

The authoritative texts are in the pinned dependency checkout:

- `LICENSE` — DeepGEMM MIT license.
- `sgl_deep_gemm/LICENSE` — SGL fork Apache 2.0 license.
- `third-party/cutlass/LICENSE.txt` — CUTLASS BSD 3-Clause license and the
  CuTeDSL EULA note.
- `third-party/fmt/LICENSE` — `{fmt}` MIT license and object-code exception.
