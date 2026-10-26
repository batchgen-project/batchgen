# DeepGEMM programmatic dependent launch (PDL).
#
# The installed DeepGEMM build ships PDL support behind a process-global
# toggle (set_pdl/get_pdl, default off). With PDL on, a DeepGEMM kernel launch
# may begin while the tail of the previous kernel is still draining; the
# kernels carry their own cudaGridDependencySynchronize() so reads of the
# previous kernel's output remain ordered. The toggle must be applied in the
# worker process before any DeepGEMM call — CUDA graph capture records the
# launch attribute, so flipping it after capture has no effect on replays.
#
# DEEPGEMM_PDL = False restores the exact pre-change behavior: apply_deepgemm_pdl
# makes no deep_gemm call at all.

DEEPGEMM_PDL = True


class DeepGemmPdlUnsupportedError(RuntimeError):
	"""The installed deep_gemm does not expose the PDL toggle."""


class DeepGemmPdlStateError(RuntimeError):
	"""deep_gemm accepted set_pdl(True) but get_pdl() does not report it."""


def apply_deepgemm_pdl() -> bool:
	"""Enable PDL on the installed deep_gemm when DEEPGEMM_PDL is set.

	Returns True when PDL was enabled, False when the switch is off.
	"""
	if not DEEPGEMM_PDL:
		return False
	import deep_gemm

	set_pdl = getattr(deep_gemm, "set_pdl", None)
	get_pdl = getattr(deep_gemm, "get_pdl", None)
	if set_pdl is None or get_pdl is None:
		raise DeepGemmPdlUnsupportedError(
			"DEEPGEMM_PDL is enabled but the installed deep_gemm exposes no "
			"set_pdl/get_pdl toggle"
		)
	set_pdl(True)
	if not get_pdl():
		raise DeepGemmPdlStateError(
			"deep_gemm.set_pdl(True) did not take effect (get_pdl() is falsy)"
		)
	return True
