"""Fixed-address device memory whose physical pages can be released and restored.

CUDA graphs and device pointer tables hold virtual addresses. A ``VmmArena``
reserves one virtual range for its lifetime and maps or unmaps physical memory
underneath it (CUDA VMM: cuMemAddressReserve / cuMemCreate / cuMemMap), so
tensors carved from it keep the same ``data_ptr()`` across ``unmap()`` and
``map()``. The memory lives outside PyTorch's caching allocator.

Contents are undefined after ``map()``; callers refill what they carved.
Touching a carved tensor while the arena is unmapped is an illegal access.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

_CARVE_ALIGN_BYTES = 256


def _round_up(value: int, multiple: int) -> int:
	return (value + multiple - 1) // multiple * multiple


def _check(result):
	"""Raise on a failed cuda.bindings driver call; return its outputs."""
	from cuda.bindings import driver as cu

	status = result[0]
	if status != cu.CUresult.CUDA_SUCCESS:
		raise RuntimeError(f"CUDA driver call failed: {status}")
	if len(result) == 1:
		return None
	return result[1] if len(result) == 2 else result[1:]


class _DeviceBytes:
	"""Minimal ``__cuda_array_interface__`` producer for a raw byte range."""

	def __init__(self, ptr: int, nbytes: int):
		self.__cuda_array_interface__ = {
			"shape": (nbytes,),
			"typestr": "|u1",
			"data": (ptr, False),
			"version": 3,
		}


class VmmArena:
	"""One reserved virtual range on one device, mapped in fixed-size chunks."""

	def __init__(
		self,
		device: torch.device,
		nbytes: int,
		chunk_bytes: int = 2 << 30,
	):
		from cuda.bindings import driver as cu

		self.device = torch.device(device)
		if self.device.type != "cuda" or self.device.index is None:
			raise ValueError(f"VmmArena needs an indexed CUDA device, got {device}")
		prop = cu.CUmemAllocationProp()
		prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
		prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
		prop.location.id = self.device.index
		granularity = int(_check(cu.cuMemGetAllocationGranularity(
			prop,
			cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
		)))
		self._prop = prop
		self._chunk_bytes = _round_up(int(chunk_bytes), granularity)
		self.nbytes = _round_up(int(nbytes), self._chunk_bytes)
		self._base = int(_check(cu.cuMemAddressReserve(self.nbytes, self._chunk_bytes, 0, 0)))
		access = cu.CUmemAccessDesc()
		access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
		access.location.id = self.device.index
		access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
		self._access = access
		self._handles = []
		self._carved_bytes = 0

	@property
	def is_mapped(self) -> bool:
		return bool(self._handles)

	def map(self) -> None:
		"""Back the whole range with fresh physical memory."""
		from cuda.bindings import driver as cu

		if self._handles:
			raise RuntimeError("VmmArena.map: already mapped")
		for offset in range(0, self.nbytes, self._chunk_bytes):
			handle = _check(cu.cuMemCreate(self._chunk_bytes, self._prop, 0))
			self._handles.append(handle)
			_check(cu.cuMemMap(self._base + offset, self._chunk_bytes, 0, handle, 0))
		_check(cu.cuMemSetAccess(self._base, self.nbytes, [self._access], 1))

	def unmap(self) -> None:
		"""Release the physical memory; the virtual range stays reserved."""
		from cuda.bindings import driver as cu

		if not self._handles:
			raise RuntimeError("VmmArena.unmap: not mapped")
		_check(cu.cuMemUnmap(self._base, self.nbytes))
		for handle in self._handles:
			_check(cu.cuMemRelease(handle))
		self._handles = []

	def carve(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
		"""Return a tensor view at the next free offset; views never move."""
		if not self._handles:
			raise RuntimeError("VmmArena.carve: map() the arena first")
		element_size = torch.empty((), dtype=dtype).element_size()
		nbytes = math.prod(shape) * element_size
		start = _round_up(self._carved_bytes, _CARVE_ALIGN_BYTES)
		if start + nbytes > self.nbytes:
			raise RuntimeError(
				f"VmmArena.carve: {nbytes} bytes at offset {start} exceed the "
				f"{self.nbytes}-byte arena"
			)
		self._carved_bytes = start + nbytes
		raw = torch.as_tensor(_DeviceBytes(self._base + start, nbytes), device=self.device)
		return raw.view(dtype).view(tuple(shape))
