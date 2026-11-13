# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#  see the license at                                                           #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
# ---------------------------------------------------------------------------- #

"""Classify the local GPU machine into one machine family.

Model planners key machine-dependent settings on the family returned here,
never on a total-memory threshold: two machines with similar HBM can differ
in host link, NVLink domain and compute rate, and a memory threshold silently
hands a new machine the constants tuned for an old one.

The signatures use the facts CUDA reports for the device (compute
capability, SM count, total memory) plus the host CPU architecture, which
separates the Grace-hosted part from the x86-hosted part of the same
Blackwell generation. A device that matches no signature raises
``UnknownMachineFamilyError``; there is no default family.
"""

from __future__ import annotations

import enum
import platform
from dataclasses import dataclass

_MIB = 1024 ** 2


class MachineFamily(str, enum.Enum):
    """GPU machine families the engine has a profile for."""

    H20 = "h20"
    H200 = "h200"
    B200 = "b200"
    B300 = "b300"
    GB300 = "gb300"


class UnknownMachineFamilyError(RuntimeError):
    """The local device matches no machine-family signature."""


@dataclass(frozen=True)
class DeviceFacts:
    """What the classifier reads from one device and its host.

    ``total_memory_bytes`` is the CUDA-reported device total
    (``cudaDeviceProp.totalGlobalMem``), ``cpu_arch`` is
    ``platform.machine()`` of the host process.
    """

    compute_capability: tuple[int, int]
    sm_count: int
    total_memory_bytes: int
    cpu_arch: str


@dataclass(frozen=True)
class FamilySignature:
    """The device facts that identify one machine family.

    Memory bands are in MiB of CUDA-reported device total and are wide
    engineering tolerances around the observed totals; they never overlap
    between families that share a compute capability and SM count.
    """

    family: MachineFamily
    compute_capabilities: frozenset[tuple[int, int]]
    sm_count: int
    memory_mib: tuple[int, int]
    cpu_archs: frozenset[str]
    gpus_per_node: int

    def matches(self, facts: DeviceFacts) -> bool:
        low, high = self.memory_mib
        memory_mib = facts.total_memory_bytes / _MIB
        return (
            tuple(facts.compute_capability) in self.compute_capabilities
            and int(facts.sm_count) == self.sm_count
            and low <= memory_mib <= high
            and facts.cpu_arch in self.cpu_archs
        )


_X86 = frozenset({"x86_64", "AMD64"})
_ARM = frozenset({"aarch64", "arm64"})

FAMILY_SIGNATURES: tuple[FamilySignature, ...] = (
    # Hopper, 78 SM, 96 GB HBM3 class.
    FamilySignature(
        MachineFamily.H20, frozenset({(9, 0)}), 78, (85_000, 120_000), _X86, 8,
    ),
    # Hopper, 132 SM, 141 GB HBM3e (CUDA reports ~139.8 GiB).
    FamilySignature(
        MachineFamily.H200, frozenset({(9, 0)}), 132, (140_000, 150_000), _X86, 8,
    ),
    # Blackwell, compute capability 10.0, 148 SM, 180 GB class.
    FamilySignature(
        MachineFamily.B200, frozenset({(10, 0)}), 148, (178_000, 190_000), _X86, 8,
    ),
    # Blackwell Ultra HGX (x86 host), 10.3, 148 SM; CUDA reports 267.68 GiB.
    FamilySignature(
        MachineFamily.B300, frozenset({(10, 3)}), 148, (270_000, 280_000), _X86, 8,
    ),
    # Blackwell Ultra on a Grace host (C2C), 10.3, 152 SM, 4 GPUs per tray.
    FamilySignature(
        MachineFamily.GB300, frozenset({(10, 3)}), 152, (280_000, 290_000), _ARM, 4,
    ),
)


def signature_for(family: MachineFamily) -> FamilySignature:
    """The signature row of ``family``."""
    for signature in FAMILY_SIGNATURES:
        if signature.family is family:
            return signature
    raise UnknownMachineFamilyError(f"no signature for machine family {family!r}")


def classify_machine_family(facts: DeviceFacts) -> MachineFamily:
    """Return the one machine family whose signature ``facts`` match.

    Raises ``UnknownMachineFamilyError`` when no signature (or, by table
    construction impossible, more than one) matches.
    """
    matches = [sig.family for sig in FAMILY_SIGNATURES if sig.matches(facts)]
    if len(matches) == 1:
        return matches[0]
    known = ", ".join(sig.family.value for sig in FAMILY_SIGNATURES)
    detail = (
        f"compute_capability={tuple(facts.compute_capability)}, "
        f"sm_count={facts.sm_count}, "
        f"total_memory={facts.total_memory_bytes / _MIB:.0f} MiB, "
        f"cpu_arch={facts.cpu_arch!r}"
    )
    if not matches:
        raise UnknownMachineFamilyError(
            f"device ({detail}) matches no machine family ({known}); add a "
            "signature and a profile row for this machine before serving on it"
        )
    raise UnknownMachineFamilyError(
        f"device ({detail}) matches several machine families "
        f"{[family.value for family in matches]}; the signatures overlap"
    )


def detect_device_facts(device: int) -> DeviceFacts:
    """Read the classifier's facts for CUDA ``device`` and this host."""
    import torch

    props = torch.cuda.get_device_properties(device)
    return DeviceFacts(
        compute_capability=(int(props.major), int(props.minor)),
        sm_count=int(props.multi_processor_count),
        total_memory_bytes=int(props.total_memory),
        cpu_arch=platform.machine(),
    )
