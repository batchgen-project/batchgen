"""Watchdog utilities for detecting stuck worker processes."""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional

import psutil

logger = logging.getLogger(__name__)


def pyspy_dump_schedulers() -> None:
    """Run py-spy dump on the current process."""
    try:
        pid = psutil.Process().pid
        cmd = f"py-spy dump --native --pid {pid}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        logger.error("Pyspy dump for PID %s:\n%s", pid, result.stdout)
    except FileNotFoundError as exc:
        logger.error("Pyspy is not available: %s", exc)
    except subprocess.CalledProcessError as exc:
        logger.error("Pyspy failed to dump PID %s. Error: %s", pid, exc.stderr)


class Watchdog:
    """Base watchdog interface."""

    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
        dump_info: Optional[Callable[[], str]] = None,
    ) -> Watchdog:
        if watchdog_timeout is None:
            assert (
                test_stuck_time == 0
            ), "stuck tester can be enabled only if watchdog is enabled."
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
            dump_info=dump_info,
        )

    def feed(self) -> None:
        pass

    @contextmanager
    def disable(self):
        yield


class _WatchdogReal(Watchdog):
    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
        dump_info: Optional[Callable[[], str]] = None,
    ) -> None:
        self._counter = 0
        self._active = True
        self._test_stuck_time = test_stuck_time
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            dump_info=dump_info,
        )
        logger.info("Watchdog %s initialized.", self._raw.debug_name)
        if self._test_stuck_time > 0:
            logger.info(
                "Watchdog %s is configured to use test_stuck_time=%s.",
                self._raw.debug_name,
                self._test_stuck_time,
            )

    def feed(self) -> None:
        if self._test_stuck_time > 0:
            logger.info(
                "Watchdog %s start deliberately stuck for %ss",
                self._raw.debug_name,
                self._test_stuck_time,
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                "Watchdog %s end deliberately stuck for %ss",
                self._raw.debug_name,
                self._test_stuck_time,
            )
        self._counter += 1

    @contextmanager
    def disable(self):
        assert self._active
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True


class _WatchdogNoop(Watchdog):
    def feed(self) -> None:
        return

    @contextmanager
    def disable(self):
        yield


class WatchdogRaw:
    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
    ) -> None:
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info

        parent = psutil.Process().parent()
        self.parent_process = parent or psutil.Process()
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self) -> None:
        try:
            while True:
                self._watchdog_once()
        except Exception as exc:
            logger.error(
                "%s watchdog thread crashed: %s",
                self.debug_name,
                exc,
                exc_info=True,
            )

    def _watchdog_once(self) -> None:
        watchdog_last_counter = 0
        watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            time.sleep(self.watchdog_timeout / 2)

        if self.dump_info is not None and (info_msg := self.dump_info()):
            logger.error("%s debug info:\n%s", self.debug_name, info_msg)

        pyspy_dump_schedulers()
        logger.error(
            "%s watchdog timeout (watchdog_timeout=%s, soft=%s)",
            self.debug_name,
            self.watchdog_timeout,
            self.soft,
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            time.sleep(5)
            try:
                self.parent_process.send_signal(signal.SIGQUIT)
            except psutil.NoSuchProcess:
                return
