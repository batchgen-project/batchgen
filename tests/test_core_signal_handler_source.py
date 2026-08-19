from pathlib import Path


SOURCE = (
    Path(__file__).parents[1] / "core" / "batchgen.cpp"
).read_text()


def test_core_signal_handler_records_sender_identity():
    assert "SA_SIGINFO" in SOURCE
    assert "[BATCHGEN_SIGNAL] signal=" in SOURCE
    assert "sender_pid=" in SOURCE
    assert "sender_uid=" in SOURCE
    assert "si_code=" in SOURCE


def test_core_signal_handler_uses_shell_conventional_exit_status():
    assert "_exit(128 + signum)" in SOURCE
