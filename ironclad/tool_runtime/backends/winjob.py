"""Windows Job Object owner (ctypes only, no pywin32).

Popen does not expose the primary thread handle, so the executor starts the child with
CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP, assigns it to the job, then resumes it with
``NtResumeProcess``. That closes the assign-after-Popen race (measured in practice: the
venv ``python.exe`` launcher spawns the real interpreter before a post-Popen assignment).
If suspension is unavailable the fallback is assign-immediately-after-Popen, which has a
tiny race for grandchildren spawned before assignment. Everything spawned after
assignment inherits the job and cannot break away (no BREAKAWAY_OK flag).
KILL_ON_JOB_CLOSE kills the tree if Nedster dies. Job Objects manage processes, not
file or network permissions.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import time
from typing import Any

from . import BackendStatus

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectBasicAccountingInformation = 1
JobObjectExtendedLimitInformation = 9
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
PROCESS_SUSPEND_RESUME = 0x0800


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801 - Win32 name
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),  # LARGE_INTEGER
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),  # DWORD
        ("MinimumWorkingSetSize", ctypes.c_size_t),  # SIZE_T
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),  # ULONG_PTR
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class IO_COUNTERS(ctypes.Structure):  # noqa: N801
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )
    ]  # fmt: skip


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):  # noqa: N801
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


_KERNEL32: Any = None


def _kernel32() -> Any:
    """A private WinDLL instance with pointer-width signatures (does not touch ctypes.windll)."""
    global _KERNEL32
    if _KERNEL32 is None:
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        handle, bool_, dword, lpvoid = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p
        signatures = {
            "CreateJobObjectW": ([lpvoid, wintypes.LPCWSTR], handle),
            "SetInformationJobObject": ([handle, ctypes.c_int, lpvoid, dword], bool_),
            "QueryInformationJobObject": (
                [handle, ctypes.c_int, lpvoid, dword, ctypes.POINTER(dword)],
                bool_,
            ),
            "AssignProcessToJobObject": ([handle, handle], bool_),
            "TerminateJobObject": ([handle, ctypes.c_uint], bool_),
            "CloseHandle": ([handle], bool_),
            "OpenProcess": ([dword, bool_, dword], handle),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(k, name)
            function.argtypes, function.restype = argtypes, restype
        _KERNEL32 = k
    return _KERNEL32


_NTDLL: Any = None


def resume_process(process: Any) -> None:
    """Resume every thread of a process created with CREATE_SUSPENDED (Popen or pid)."""
    global _NTDLL
    if _NTDLL is None:
        from ctypes import wintypes

        ntdll = ctypes.WinDLL("ntdll")  # type: ignore[attr-defined]
        ntdll.NtResumeProcess.argtypes, ntdll.NtResumeProcess.restype = [wintypes.HANDLE], ctypes.c_long
        _NTDLL = ntdll
    k = _kernel32()
    raw = getattr(process, "_handle", None)
    opened = None
    if raw is None:
        pid = process if isinstance(process, int) else process.pid
        opened = k.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not opened:
            raise _error("OpenProcess")
        raw = opened
    try:
        status = _NTDLL.NtResumeProcess(int(raw))
        if status < 0:
            raise OSError(status, f"NtResumeProcess failed (NTSTATUS {status & 0xFFFFFFFF:#x})")
    finally:
        if opened:
            k.CloseHandle(opened)


def _error(what: str) -> OSError:
    code = ctypes.get_last_error()  # type: ignore[attr-defined]
    return OSError(code, f"{what} failed (Win32 error {code})")


def probe() -> BackendStatus:
    if os.name != "nt":
        return BackendStatus("winjob", False, "process-only", "Windows only")
    try:
        job = WinJob()
        job.close()
    except OSError as exc:
        return BackendStatus("winjob", False, "process-only", str(exc))
    return BackendStatus(
        "winjob", True, "process-only", "Job Object: tree kill only; no file/network isolation"
    )


class WinJob:
    """One Job Object with kill-on-close. Always ``close()`` (also done by ``__del__``)."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Job Objects are Windows only")
        self._k = _kernel32()
        self.handle: int | None = self._k.CreateJobObjectW(None, None)
        if not self.handle:
            self.handle = None
            raise _error("CreateJobObjectW")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = self._k.SetInformationJobObject(
            self.handle, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            error = _error("SetInformationJobObject")
            self.close()
            raise error

    def assign(self, process: Any) -> None:
        """Assign a ``subprocess.Popen`` (uses its process handle) or a raw pid."""
        if self.handle is None:
            raise OSError("job is closed")
        raw = getattr(process, "_handle", None)
        opened = None
        if raw is None:
            pid = process if isinstance(process, int) else process.pid
            opened = self._k.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
            if not opened:
                raise _error("OpenProcess")
            raw = opened
        try:
            if not self._k.AssignProcessToJobObject(self.handle, int(raw)):
                raise _error("AssignProcessToJobObject")
        finally:
            if opened:
                self._k.CloseHandle(opened)

    def active_processes(self) -> int:
        if self.handle is None:
            return 0
        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        ok = self._k.QueryInformationJobObject(
            self.handle, JobObjectBasicAccountingInformation, ctypes.byref(info), ctypes.sizeof(info), None
        )
        if not ok:
            raise _error("QueryInformationJobObject")
        return int(info.ActiveProcesses)

    def terminate(self, wait_s: float = 5.0) -> bool:
        """Kill every process in the job; return True once the job reports none active."""
        if self.handle is None:
            return True
        self._k.TerminateJobObject(self.handle, 1)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                if self.active_processes() == 0:
                    return True
            except OSError:
                return False
            time.sleep(0.02)
        return False

    def close(self) -> None:
        handle, self.handle = self.handle, None
        if handle:
            self._k.CloseHandle(handle)

    def __del__(self) -> None:  # pragma: no cover - safety net
        with contextlib.suppress(Exception):
            self.close()
