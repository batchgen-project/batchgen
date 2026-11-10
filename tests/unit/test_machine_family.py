"""Machine-family classifier: one family per observed device, named error
otherwise."""

import pytest

from batchgen.hardware.machine_family import (
    FAMILY_SIGNATURES,
    DeviceFacts,
    MachineFamily,
    UnknownMachineFamilyError,
    classify_machine_family,
    signature_for,
)

GIB = 1024 ** 3
MIB = 1024 ** 2

# CUDA-reported facts observed on each family (memory is the device total
# torch reports, which sits slightly below the nvidia-smi figure).
OBSERVED = {
    MachineFamily.H20: [
        DeviceFacts((9, 0), 78, int(89.05 * GIB), "x86_64"),
        DeviceFacts((9, 0), 78, 97_871 * MIB, "x86_64"),
    ],
    MachineFamily.H200: [DeviceFacts((9, 0), 132, int(139.81 * GIB), "x86_64")],
    MachineFamily.B200: [DeviceFacts((10, 0), 148, 182_640 * MIB, "x86_64")],
    MachineFamily.B300: [DeviceFacts((10, 3), 148, int(267.68 * GIB), "x86_64")],
    MachineFamily.GB300: [DeviceFacts((10, 3), 152, 283_270 * MIB, "aarch64")],
}


@pytest.mark.parametrize(
    ("family", "facts"),
    [(family, facts) for family, rows in OBSERVED.items() for facts in rows],
)
def test_each_observed_device_classifies_to_its_family(family, facts):
    assert classify_machine_family(facts) is family


def test_every_family_has_exactly_one_signature():
    assert sorted(sig.family.value for sig in FAMILY_SIGNATURES) == sorted(
        family.value for family in MachineFamily
    )
    assert signature_for(MachineFamily.GB300).gpus_per_node == 4
    assert signature_for(MachineFamily.B300).gpus_per_node == 8


@pytest.mark.parametrize(
    "facts",
    [
        # Hopper 132 SM with 80 GB: neither H20 nor H200
        DeviceFacts((9, 0), 132, 80 * GIB, "x86_64"),
        # H20 memory with H200 SM count
        DeviceFacts((9, 0), 132, int(89.05 * GIB), "x86_64"),
        # B300-sized device reporting the GB300 SM count on x86
        DeviceFacts((10, 3), 152, int(267.68 * GIB), "x86_64"),
        # GB300 facts on an x86 host
        DeviceFacts((10, 3), 152, 283_270 * MIB, "x86_64"),
        # B300 facts on an Arm host
        DeviceFacts((10, 3), 148, int(267.68 * GIB), "aarch64"),
        # Blackwell 10.0 with B300 memory
        DeviceFacts((10, 0), 148, int(267.68 * GIB), "x86_64"),
        # an Ampere part
        DeviceFacts((8, 0), 108, 80 * GIB, "x86_64"),
        # a memory total outside every band
        DeviceFacts((10, 3), 148, 200 * GIB, "x86_64"),
    ],
)
def test_unknown_device_raises_the_named_error(facts):
    with pytest.raises(UnknownMachineFamilyError, match="matches no machine family"):
        classify_machine_family(facts)


def test_signatures_never_overlap_on_the_observed_devices():
    for rows in OBSERVED.values():
        for facts in rows:
            assert sum(sig.matches(facts) for sig in FAMILY_SIGNATURES) == 1
