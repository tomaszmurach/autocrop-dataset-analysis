"""Opt-in research profiling for the exact candidate-retrieval baseline."""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os
import platform
import sys
from time import perf_counter_ns
from typing import Callable


PROFILING_SCHEMA_VERSION = "1.0"
PROFILING_SEMANTICS = "RESEARCH_ONLY_EXACT_BF_BASELINE_PROFILING"


@dataclass(frozen=True, slots=True)
class StageMeasurement:
    stage: str
    elapsed_ns: int
    work_counts: tuple[tuple[str, int], ...] = ()
    item_ordinal: int | None = None
    phase: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessMemoryReading:
    provider: str
    current_rss_bytes: int | None
    peak_rss_bytes: int | None


@dataclass(frozen=True, slots=True)
class MemoryMeasurement:
    label: str
    provider: str
    current_rss_bytes: int | None
    peak_rss_bytes: int | None


class _StageTimer(AbstractContextManager["_StageTimer"]):
    def __init__(
        self,
        profiler: RetrievalProfiler | None,
        stage: str,
        *,
        item_ordinal: int | None,
        phase: str | None,
    ) -> None:
        self._profiler = profiler
        self._stage = stage
        self._item_ordinal = item_ordinal
        self._phase = phase
        self._started_ns: int | None = None
        self._work_counts: dict[str, int] = {}

    def __enter__(self) -> _StageTimer:
        if self._profiler is not None:
            self._started_ns = perf_counter_ns()
        return self

    def add_work(self, **counts: int) -> None:
        if self._profiler is None:
            return
        for name, value in counts.items():
            if type(value) is not int or value < 0:
                raise ValueError("INVALID_PROFILING_WORK_COUNT")
            self._work_counts[name] = self._work_counts.get(name, 0) + value

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._profiler is not None and self._started_ns is not None:
            self._profiler._append_measurement(
                StageMeasurement(
                    stage=self._stage,
                    elapsed_ns=max(0, perf_counter_ns() - self._started_ns),
                    work_counts=tuple(sorted(self._work_counts.items())),
                    item_ordinal=self._item_ordinal,
                    phase=self._phase,
                )
            )
        return None


class RetrievalProfiler:
    """Collect coarse timings and process-memory snapshots for retrieval only."""

    def __init__(
        self,
        *,
        memory_reader: Callable[[], ProcessMemoryReading] | None = None,
    ) -> None:
        self._measurements: list[StageMeasurement] = []
        self._memory: list[MemoryMeasurement] = []
        self._memory_reader = memory_reader or read_process_memory

    @property
    def measurements(self) -> tuple[StageMeasurement, ...]:
        return tuple(self._measurements)

    @property
    def memory_measurements(self) -> tuple[MemoryMeasurement, ...]:
        return tuple(self._memory)

    def measure(
        self,
        stage: str,
        *,
        item_ordinal: int | None = None,
        phase: str | None = None,
    ) -> _StageTimer:
        if not stage:
            raise ValueError("PROFILING_STAGE_REQUIRED")
        if item_ordinal is not None and (type(item_ordinal) is not int or item_ordinal < 0):
            raise ValueError("INVALID_PROFILING_ITEM_ORDINAL")
        return _StageTimer(
            self,
            stage,
            item_ordinal=item_ordinal,
            phase=phase,
        )

    def snapshot_memory(self, label: str) -> MemoryMeasurement:
        if not label:
            raise ValueError("PROFILING_MEMORY_LABEL_REQUIRED")
        reading = self._memory_reader()
        measurement = MemoryMeasurement(
            label=label,
            provider=reading.provider,
            current_rss_bytes=reading.current_rss_bytes,
            peak_rss_bytes=reading.peak_rss_bytes,
        )
        self._memory.append(measurement)
        return measurement

    def elapsed_ns(self, stage: str, *, phase: str | None = None) -> int:
        return sum(
            measurement.elapsed_ns
            for measurement in self._measurements
            if measurement.stage == stage and measurement.phase == phase
        )

    def as_report(self) -> dict[str, object]:
        grouped: dict[tuple[str, str | None], list[StageMeasurement]] = defaultdict(list)
        for measurement in self._measurements:
            grouped[(measurement.stage, measurement.phase)].append(measurement)

        stages: list[dict[str, object]] = []
        for (stage, phase), measurements in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1] or "")
        ):
            elapsed = [measurement.elapsed_ns for measurement in measurements]
            work: dict[str, int] = defaultdict(int)
            for measurement in measurements:
                for name, value in measurement.work_counts:
                    work[name] += value
            stage_record: dict[str, object] = {
                "stage": stage,
                "invocation_count": len(measurements),
                "elapsed_ns_total": sum(elapsed),
                "elapsed_ns_min": min(elapsed),
                "elapsed_ns_max": max(elapsed),
                "work_counts": dict(sorted(work.items())),
            }
            if phase is not None:
                stage_record["phase"] = phase
            stages.append(stage_record)

        per_item = []
        for measurement in self._measurements:
            if measurement.item_ordinal is None:
                continue
            item = {
                "stage": measurement.stage,
                "item_ordinal": measurement.item_ordinal,
                "elapsed_ns": measurement.elapsed_ns,
                "work_counts": dict(measurement.work_counts),
            }
            if measurement.phase is not None:
                item["phase"] = measurement.phase
            per_item.append(item)

        return {
            "schema_version": PROFILING_SCHEMA_VERSION,
            "semantics": PROFILING_SEMANTICS,
            "timing": {
                "clock": "time.perf_counter_ns",
                "duration_unit": "nanoseconds",
                "filesystem_page_cache": "UNCONTROLLED",
            },
            "stages": stages,
            "per_item": per_item,
            "memory": {
                "snapshots": [
                    {
                        "label": measurement.label,
                        "provider": measurement.provider,
                        "current_rss_bytes": measurement.current_rss_bytes,
                        "peak_rss_bytes": measurement.peak_rss_bytes,
                    }
                    for measurement in self._memory
                ]
            },
        }

    def _append_measurement(self, measurement: StageMeasurement) -> None:
        self._measurements.append(measurement)


def profiling_stage(
    profiler: RetrievalProfiler | None,
    stage: str,
    *,
    item_ordinal: int | None = None,
    phase: str | None = None,
) -> _StageTimer:
    """Return a timer that is inert when profiling is disabled."""

    return _StageTimer(
        profiler,
        stage,
        item_ordinal=item_ordinal,
        phase=phase,
    )


def read_process_memory() -> ProcessMemoryReading:
    """Read process RSS/peak RSS without substituting Python heap metrics."""

    if sys.platform == "win32":
        return _read_windows_process_memory()
    return _read_posix_process_memory()


def _read_windows_process_memory() -> ProcessMemoryReading:
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        )

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error())
        return ProcessMemoryReading(
            provider="windows_psapi_working_set",
            current_rss_bytes=int(counters.WorkingSetSize),
            peak_rss_bytes=int(counters.PeakWorkingSetSize),
        )
    except Exception:
        return ProcessMemoryReading("unavailable", None, None)


def _read_posix_process_memory() -> ProcessMemoryReading:
    try:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_bytes = maximum if platform.system() == "Darwin" else maximum * 1024
        current_bytes = _read_linux_current_rss()
        return ProcessMemoryReading(
            provider="resource.getrusage",
            current_rss_bytes=current_bytes,
            peak_rss_bytes=peak_bytes,
        )
    except Exception:
        return ProcessMemoryReading("unavailable", None, None)


def _read_linux_current_rss() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/statm", encoding="ascii") as source:
            resident_pages = int(source.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None
