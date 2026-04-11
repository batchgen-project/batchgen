"""Unit tests for batchgen.worker.config.WorkerConfig."""

from __future__ import annotations

import pytest

from batchgen.worker.config import WorkerConfig


class TestDefaults:
    def test_defaults_with_no_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every BATCHGEN_* env var unset → production defaults hold."""
        for key in list(monkeypatch._setitem):  # no-op hint
            pass
        # Clear any ambient BATCHGEN_* env so the factory gets a clean slate
        for key in list(__import__("os").environ.keys()):
            if key.startswith("BATCHGEN_"):
                monkeypatch.delenv(key, raising=False)

        cfg = WorkerConfig.from_env(
            host_kv_total_pages=12345,
            model_context_length=4096,
        )

        assert cfg.decision_frequency_pages == 2
        assert cfg.initial_gpu_page_buffer == 32
        assert cfg.extension_gpu_page_buffer == 4
        assert cfg.prefill_watermark_pct == 70
        assert cfg.eviction_watermark_pct == 10
        assert cfg.host_kv_total_pages == 12345  # injected
        assert cfg.model_context_length == 4096  # injected
        assert cfg.rep_detection_enabled is True
        assert cfg.preemption_enabled is True
        assert cfg.ignore_eos is False
        assert cfg.max_pool_size == 0
        assert cfg.decode_assert is False
        assert cfg.multi_batch_diag is False
        assert cfg.decode_timing is False
        assert cfg.critical_diags is False
        assert cfg.cb_log == ""


class TestIntegerOverrides:
    def test_watermarks_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BATCHGEN_HOST_KV_WATERMARK", "80")
        monkeypatch.setenv("BATCHGEN_HOST_KV_EVICTION_WATERMARK", "15")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.prefill_watermark_pct == 80
        assert cfg.eviction_watermark_pct == 15

    def test_decision_frequency_pages_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BATCHGEN_DECISION_FREQUENCY_PAGES", "4")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.decision_frequency_pages == 4

    def test_invalid_int_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BATCHGEN_HOST_KV_WATERMARK", "not-an-int")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.prefill_watermark_pct == 70  # default


class TestBooleanOverrides:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("FALSE", False),
            ("no", False),
            ("off", False),
            ("", False),
        ],
    )
    def test_bool_parsing_variants(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        monkeypatch.setenv("BATCHGEN_REP_DETECTION", raw)
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.rep_detection_enabled is expected

    def test_invalid_bool_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typos like BATCHGEN_REP_DETECTION=maybe fall back to the
        production default instead of raising — handlers stay robust."""
        monkeypatch.setenv("BATCHGEN_REP_DETECTION", "maybe")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.rep_detection_enabled is True  # default

    def test_decode_preemption_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BATCHGEN_ENABLE_DECODE_PREEMPTION", "0")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.preemption_enabled is False

    def test_decode_assert_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BATCHGEN_DECODE_ASSERT", "1")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.decode_assert is True


class TestStringKnob:
    def test_cb_log_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BATCHGEN_CB_LOG", "boundary-debug")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.cb_log == "boundary-debug"


class TestInjectedFields:
    def test_host_kv_total_pages_is_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=999_999, model_context_length=4096
        )
        assert cfg.host_kv_total_pages == 999_999

    def test_model_context_length_is_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=131_072
        )
        assert cfg.model_context_length == 131_072

    def test_max_pool_size_is_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000,
            model_context_length=4096,
            max_pool_size=10240,
        )
        assert cfg.max_pool_size == 10240


class TestRawEnvSnapshot:
    def test_raw_env_captures_only_batchgen_prefixed_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BATCHGEN_REP_DETECTION", "0")
        monkeypatch.setenv("BATCHGEN_DECODE_TIMING", "1")
        monkeypatch.setenv("NOT_A_BATCHGEN_VAR", "shouldnotappear")
        cfg = WorkerConfig.from_env(
            host_kv_total_pages=1000, model_context_length=4096
        )
        assert cfg.raw_env.get("BATCHGEN_REP_DETECTION") == "0"
        assert cfg.raw_env.get("BATCHGEN_DECODE_TIMING") == "1"
        assert "NOT_A_BATCHGEN_VAR" not in cfg.raw_env


class TestFrozenness:
    def test_config_is_frozen(self) -> None:
        cfg = WorkerConfig(host_kv_total_pages=100, model_context_length=4096)
        with pytest.raises(Exception):
            cfg.prefill_watermark_pct = 99  # type: ignore[misc]
