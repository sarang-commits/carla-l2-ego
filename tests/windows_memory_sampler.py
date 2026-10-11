"""Isolated Mission 4G.6 Windows process-memory sampler.

This module is deliberately self-contained and stdlib-only.  Importing it
performs no native work: every kernel32/psapi/pdh binding is created lazily
inside the real backend constructors, so the module is importable and fully
testable on non-live paths with injected fake backends and without any CARLA
process.

Process identity comes exclusively from the selected PID and the executable
image reported by the opened handle; no installation path is assumed.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum

SHIPPING_PROCESS_IMAGE_NAME = "CarlaUE4-Win64-Shipping.exe"

# -- Win32 constants -----------------------------------------------------

TH32CS_SNAPPROCESS = 0x00000002
ERROR_NO_MORE_FILES = 18
PROCESS_QUERY_LIMITED_INFORMATION = 0x00001000
SYNCHRONIZE = 0x00100000
PROCESS_OPEN_ACCESS = PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF
MAX_PROCESS_NAME_CHARACTERS = 260
MAX_IMAGE_PATH_CHARACTERS = 32767
MAX_NATIVE_ERROR_EVIDENCE_CHARACTERS = 512
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2
PDH_NO_DATA = 0x800007D5
PDH_CSTATUS_VALID_DATA = 0x00000000
PDH_CSTATUS_NEW_DATA = 0x00000001
GPU_PROCESS_MEMORY_COUNTER_PATH = "\\GPU Process Memory(*)\\Dedicated Usage"


def _add_bounded_secondary_note(
    primary: BaseException, context: str, secondary: BaseException
) -> None:
    """Attach bounded cleanup evidence without replacing the primary."""
    try:
        detail = str(secondary)
    except BaseException as conversion_error:
        detail = f"<unprintable: {type(conversion_error).__name__}>"
    evidence = (
        f"{context}: {type(secondary).__name__}: {detail}"
    )[:MAX_NATIVE_ERROR_EVIDENCE_CHARACTERS]
    primary.add_note(evidence)


# -- Native structures ---------------------------------------------------


class FILETIME(ctypes.Structure):
    _fields_ = (
        ("dwLowDateTime", ctypes.c_uint32),
        ("dwHighDateTime", ctypes.c_uint32),
    )


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = (
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_int32),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * MAX_PROCESS_NAME_CHARACTERS),
    )


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
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


class _PDH_FMT_COUNTERVALUE_UNION(ctypes.Union):
    _fields_ = (
        ("longValue", ctypes.c_int32),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("AnsiStringValue", ctypes.c_char_p),
        ("WideStringValue", ctypes.c_wchar_p),
    )


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = (
        ("CStatus", ctypes.c_uint32),
        ("value", _PDH_FMT_COUNTERVALUE_UNION),
    )


class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = (
        ("szName", ctypes.c_wchar_p),
        ("FmtValue", PDH_FMT_COUNTERVALUE),
    )


# -- Errors and value objects --------------------------------------------


class WindowsMemorySamplerError(Exception):
    """Base error for every failure raised by this module."""


class ProcessResolutionError(WindowsMemorySamplerError):
    """The shipping process could not be uniquely and exactly resolved."""


class ProcessAccessError(WindowsMemorySamplerError):
    """The resolved process could not be opened or identified."""


class ProcessExitedError(WindowsMemorySamplerError):
    """The original opened process object has exited."""


class CounterReadError(WindowsMemorySamplerError):
    """A memory-counter read against the retained handle failed."""


class PdhQueryError(WindowsMemorySamplerError):
    """A PDH call failed with a nonzero status."""


class PdhNoDataError(PdhQueryError):
    """PDH reported that no counter data exists for the query."""


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Immutable identity of the exact opened process object."""

    pid: int
    image_path: str
    creation_time_100ns: int


@dataclass(frozen=True, slots=True)
class ProcessMemoryCounters:
    """One PROCESS_MEMORY_COUNTERS_EX reading copied to plain integers.

    ``pagefile_usage_bytes`` is retained only as an optional
    alias-consistency diagnostic against ``private_commit_bytes``; it is
    never an independent third metric.
    """

    working_set_bytes: int
    private_commit_bytes: int
    pagefile_usage_bytes: int


class GpuAvailability(Enum):
    AVAILABLE = "AVAILABLE"
    UNSUPPORTED = "UNSUPPORTED"
    NO_ACTIVITY = "NO_ACTIVITY"
    ZERO_OR_UNIFIED_MEMORY = "ZERO_OR_UNIFIED_MEMORY"
    UNRELIABLE = "UNRELIABLE"


@dataclass(frozen=True, slots=True)
class GpuDedicatedSample:
    """One advisory GPU dedicated-memory observation."""

    status: GpuAvailability
    dedicated_bytes: int | None
    instance_count: int


def counters_from_memory_structure(structure) -> ProcessMemoryCounters:
    """Map one PROCESS_MEMORY_COUNTERS_EX structure to plain integers."""
    working_set = int(structure.WorkingSetSize)
    private_commit = int(structure.PrivateUsage)
    pagefile = int(structure.PagefileUsage)
    for name, value in (
        ("WorkingSetSize", working_set),
        ("PrivateUsage", private_commit),
        ("PagefileUsage", pagefile),
    ):
        if value < 0:
            raise CounterReadError(f"{name} must be non-negative, got {value}")
    return ProcessMemoryCounters(
        working_set_bytes=working_set,
        private_commit_bytes=private_commit,
        pagefile_usage_bytes=pagefile,
    )


def filetime_to_100ns(filetime) -> int:
    """Convert one FILETIME to a single 64-bit 100-nanosecond integer."""
    return (int(filetime.dwHighDateTime) << 32) | int(filetime.dwLowDateTime)


def image_base_name(image_path: str) -> str:
    """Return the final path component of a Windows image path."""
    return image_path.replace("/", "\\").rsplit("\\", 1)[-1]


# -- Real process backend ------------------------------------------------


class WindowsNativeBackend:
    """Late-bound kernel32/psapi bindings for the process sampler.

    All DLL loading happens here, at construction time, never at module
    import time.  Every method either returns plain Python values or raises
    OSError; typed classification is the sampler's responsibility.
    """

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        try:
            get_memory_info = kernel32.K32GetProcessMemoryInfo
        except AttributeError:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_memory_info = psapi.GetProcessMemoryInfo

        kernel32.CreateToolhelp32Snapshot.argtypes = (
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        kernel32.Process32FirstW.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(PROCESSENTRY32W),
        )
        kernel32.Process32FirstW.restype = ctypes.c_int32
        kernel32.Process32NextW.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(PROCESSENTRY32W),
        )
        kernel32.Process32NextW.restype = ctypes.c_int32
        kernel32.OpenProcess.argtypes = (
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
        )
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int32
        kernel32.QueryFullProcessImageNameW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_uint32),
        )
        kernel32.QueryFullProcessImageNameW.restype = ctypes.c_int32
        kernel32.GetProcessTimes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        )
        kernel32.GetProcessTimes.restype = ctypes.c_int32
        kernel32.WaitForSingleObject.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        get_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            ctypes.c_uint32,
        )
        get_memory_info.restype = ctypes.c_int32

        self._kernel32 = kernel32
        self._get_memory_info = get_memory_info

    def _last_error(self, call: str) -> OSError:
        code = ctypes.get_last_error()
        return OSError(f"{call} failed with Windows error {code}")

    def snapshot_processes(self) -> tuple[tuple[int, str], ...]:
        snapshot = self._kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot is None or snapshot == INVALID_HANDLE_VALUE:
            raise self._last_error("CreateToolhelp32Snapshot")
        primary_failure = None
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            entries = []
            proceeded = self._kernel32.Process32FirstW(
                snapshot, ctypes.byref(entry)
            )
            while proceeded:
                entries.append((int(entry.th32ProcessID), str(entry.szExeFile)))
                proceeded = self._kernel32.Process32NextW(
                    snapshot, ctypes.byref(entry)
                )
            code = ctypes.get_last_error()
            if code != ERROR_NO_MORE_FILES:
                raise OSError(
                    f"Toolhelp enumeration failed with Windows error {code}"
                )
            return tuple(entries)
        except BaseException as exc:
            primary_failure = exc
            raise
        finally:
            try:
                # Snapshot closure is mandatory process-discovery work.  Use
                # the same checked production path as retained process handles
                # so a failed BOOL result can never be mistaken for success.
                self.close_handle(snapshot)
            except BaseException as close_exc:
                if primary_failure is None:
                    raise
                # Enumeration remains the primary failure.  Exception notes
                # retain bounded close evidence without replacing its type,
                # identity, traceback, or already-captured GetLastError text.
                evidence = (
                    "Toolhelp snapshot close also failed: "
                    f"{type(close_exc).__name__}: {close_exc}"
                )[:MAX_NATIVE_ERROR_EVIDENCE_CHARACTERS]
                primary_failure.add_note(evidence)

    def open_process(self, pid: int):
        handle = self._kernel32.OpenProcess(PROCESS_OPEN_ACCESS, 0, pid)
        if not handle:
            raise self._last_error(f"OpenProcess({pid})")
        return handle

    def close_handle(self, handle) -> None:
        if not self._kernel32.CloseHandle(handle):
            raise self._last_error("CloseHandle")

    def query_image_path(self, handle) -> str:
        size = ctypes.c_uint32(MAX_IMAGE_PATH_CHARACTERS)
        buffer = ctypes.create_unicode_buffer(MAX_IMAGE_PATH_CHARACTERS)
        succeeded = self._kernel32.QueryFullProcessImageNameW(
            handle, 0, buffer, ctypes.byref(size)
        )
        if not succeeded:
            raise self._last_error("QueryFullProcessImageNameW")
        return buffer.value

    def query_creation_time(self, handle) -> int:
        creation = FILETIME()
        exit_time = FILETIME()
        kernel_time = FILETIME()
        user_time = FILETIME()
        succeeded = self._kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not succeeded:
            raise self._last_error("GetProcessTimes")
        return filetime_to_100ns(creation)

    def wait_zero(self, handle) -> int:
        return int(self._kernel32.WaitForSingleObject(handle, 0))

    def read_memory_counters(self, handle) -> ProcessMemoryCounters:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        succeeded = self._get_memory_info(
            handle, ctypes.byref(counters), counters.cb
        )
        if not succeeded:
            raise self._last_error("GetProcessMemoryInfo")
        return counters_from_memory_structure(counters)


# -- Process sampler -----------------------------------------------------


class WindowsProcessMemorySampler:
    """Sampler pinned to exactly one opened shipping-process object.

    The one process handle opened by :meth:`open` is retained for the whole
    sampler lifetime.  Because a handle refers to the process object rather
    than the numeric PID, a recycled PID can never be observed as the
    original process, and this sampler never reacquires or follows any
    replacement process.  Exit detection uses ``WaitForSingleObject`` with a
    zero timeout against that same retained handle.
    """

    __slots__ = ("_backend", "_closed", "_handle", "_identity")

    def __init__(self, backend, handle, identity: ProcessIdentity) -> None:
        if not isinstance(identity, ProcessIdentity):
            raise TypeError("identity must be ProcessIdentity")
        self._backend = backend
        self._handle = handle
        self._identity = identity
        self._closed = False

    @classmethod
    def open(cls, *, pid: int | None = None, backend=None):
        """Resolve, verify, and open exactly one shipping process."""
        if pid is not None and (
            isinstance(pid, bool) or type(pid) is not int or pid < 1
        ):
            raise ProcessResolutionError(
                "explicit pid must be a positive integer"
            )
        active_backend = backend if backend is not None else WindowsNativeBackend()

        try:
            entries = active_backend.snapshot_processes()
        except WindowsMemorySamplerError:
            raise
        except Exception as exc:
            raise ProcessResolutionError(
                f"process enumeration failed: {type(exc).__name__}: {exc}"
            ) from exc

        expected = SHIPPING_PROCESS_IMAGE_NAME.lower()
        matches = tuple(
            entry_pid
            for entry_pid, entry_name in entries
            if isinstance(entry_name, str) and entry_name.lower() == expected
        )
        if pid is not None:
            known_pids = {entry_pid for entry_pid, _entry_name in entries}
            if pid not in known_pids:
                raise ProcessResolutionError(
                    f"configured pid {pid} is not a running process"
                )
            if pid not in matches:
                raise ProcessResolutionError(
                    f"configured pid {pid} is not "
                    f"{SHIPPING_PROCESS_IMAGE_NAME}"
                )
            selected = pid
        else:
            if not matches:
                raise ProcessResolutionError(
                    f"no {SHIPPING_PROCESS_IMAGE_NAME} process is running"
                )
            if len(matches) > 1:
                raise ProcessResolutionError(
                    f"{len(matches)} {SHIPPING_PROCESS_IMAGE_NAME} processes "
                    "are running; select one with CARLA_SERVER_PID"
                )
            selected = matches[0]

        try:
            handle = active_backend.open_process(selected)
        except WindowsMemorySamplerError:
            raise
        except Exception as exc:
            raise ProcessAccessError(
                f"opening pid {selected} failed: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            image_path = active_backend.query_image_path(handle)
            if (
                not isinstance(image_path, str)
                or image_base_name(image_path).lower() != expected
            ):
                raise ProcessResolutionError(
                    f"opened pid {selected} image is not "
                    f"{SHIPPING_PROCESS_IMAGE_NAME}"
                )
            creation_time = active_backend.query_creation_time(handle)
            if type(creation_time) is not int or creation_time < 0:
                raise ProcessAccessError(
                    "process creation time must be a non-negative integer"
                )
        except BaseException as failure:
            if isinstance(failure, WindowsMemorySamplerError) or not isinstance(
                failure, Exception
            ):
                primary = failure
            else:
                primary = ProcessAccessError(
                    f"identifying pid {selected} failed: "
                    f"{type(failure).__name__}: {failure}"
                )
            cls._close_checked_after_open_failure(
                active_backend, handle, primary
            )
            if primary is failure:
                raise
            raise primary from failure

        identity = ProcessIdentity(
            pid=selected,
            image_path=image_path,
            creation_time_100ns=creation_time,
        )
        return cls(active_backend, handle, identity)

    @staticmethod
    def _close_checked_after_open_failure(
        backend, handle, primary: BaseException
    ) -> None:
        """Close one locally owned handle and retain checked failure evidence."""
        try:
            backend.close_handle(handle)
        except BaseException as close_failure:
            _add_bounded_secondary_note(
                primary,
                "process handle rollback close also failed",
                close_failure,
            )

    @property
    def identity(self) -> ProcessIdentity:
        return self._identity

    @property
    def closed(self) -> bool:
        return self._closed

    def _wait_state(self) -> int:
        try:
            return int(self._backend.wait_zero(self._handle))
        except Exception as exc:
            raise CounterReadError(
                f"WaitForSingleObject failed: {type(exc).__name__}: {exc}"
            ) from exc

    def has_exited(self) -> bool:
        """Report whether the original opened process object has exited."""
        if self._closed:
            raise CounterReadError("sampler is closed")
        return self._wait_state() == WAIT_OBJECT_0

    def sample(self) -> ProcessMemoryCounters:
        """Take exactly one counter reading; never retried internally."""
        if self._closed:
            raise CounterReadError("sampler is closed")
        wait_state = self._wait_state()
        if wait_state == WAIT_OBJECT_0:
            raise ProcessExitedError(
                f"process {self._identity.pid} has exited"
            )
        if wait_state != WAIT_TIMEOUT:
            raise CounterReadError(
                f"WaitForSingleObject returned unexpected state {wait_state}"
            )
        try:
            counters = self._backend.read_memory_counters(self._handle)
        except WindowsMemorySamplerError:
            raise
        except Exception as exc:
            raise CounterReadError(
                f"memory counter read failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(counters, ProcessMemoryCounters):
            raise CounterReadError(
                "backend must return ProcessMemoryCounters"
            )
        return counters

    @staticmethod
    def pagefile_alias_consistent(counters: ProcessMemoryCounters) -> bool:
        """Advisory PagefileUsage/PrivateUsage alias-consistency check."""
        if not isinstance(counters, ProcessMemoryCounters):
            raise TypeError("counters must be ProcessMemoryCounters")
        return counters.pagefile_usage_bytes == counters.private_commit_bytes

    def close(self) -> None:
        """Close the retained native handle exactly once."""
        if self._closed:
            return
        self._closed = True
        self._backend.close_handle(self._handle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# -- Real PDH backend ----------------------------------------------------


class WindowsPdhBackend:
    """Late-bound pdh.dll bindings for advisory GPU evidence."""

    def __init__(self) -> None:
        pdh = ctypes.WinDLL("pdh")
        try:
            add_counter = pdh.PdhAddEnglishCounterW
        except AttributeError:
            add_counter = pdh.PdhAddCounterW

        pdh.PdhOpenQueryW.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )
        pdh.PdhOpenQueryW.restype = ctypes.c_uint32
        add_counter.argtypes = (
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )
        add_counter.restype = ctypes.c_uint32
        pdh.PdhCollectQueryData.argtypes = (ctypes.c_void_p,)
        pdh.PdhCollectQueryData.restype = ctypes.c_uint32
        pdh.PdhGetFormattedCounterArrayW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        )
        pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_uint32
        pdh.PdhCloseQuery.argtypes = (ctypes.c_void_p,)
        pdh.PdhCloseQuery.restype = ctypes.c_uint32

        self._pdh = pdh
        self._add_counter = add_counter

    @staticmethod
    def _status_text(status: int) -> str:
        return f"0x{status & 0xFFFFFFFF:08X}"

    def open_query(self):
        query = ctypes.c_void_p()
        status = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(query))
        if status != 0:
            raise PdhQueryError(
                f"PdhOpenQueryW failed with {self._status_text(status)}"
            )
        return query

    def add_english_counter(self, query, path: str):
        counter = ctypes.c_void_p()
        status = self._add_counter(query, path, 0, ctypes.byref(counter))
        if status != 0:
            raise PdhQueryError(
                f"adding counter {path!r} failed with "
                f"{self._status_text(status)}"
            )
        return counter

    def collect(self, query) -> None:
        status = int(self._pdh.PdhCollectQueryData(query))
        if status == PDH_NO_DATA:
            raise PdhNoDataError("PdhCollectQueryData reported no data")
        if status != 0:
            raise PdhQueryError(
                f"PdhCollectQueryData failed with {self._status_text(status)}"
            )

    def formatted_counter_array(
        self, counter
    ) -> tuple[tuple[str | None, int, int], ...]:
        # The two PdhGetFormattedCounterArrayW calls below are the documented
        # buffer-sizing protocol of one logical read, not a retried sample.
        buffer_size = ctypes.c_uint32(0)
        item_count = ctypes.c_uint32(0)
        status = int(
            self._pdh.PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_LARGE,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                None,
            )
        )
        if status == PDH_NO_DATA:
            raise PdhNoDataError(
                "PdhGetFormattedCounterArrayW reported no data"
            )
        if status == 0 and item_count.value == 0:
            return ()
        if status != PDH_MORE_DATA:
            raise PdhQueryError(
                "PdhGetFormattedCounterArrayW sizing failed with "
                f"{self._status_text(status)}"
            )
        raw = (ctypes.c_ubyte * buffer_size.value)()
        status = int(
            self._pdh.PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_LARGE,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                raw,
            )
        )
        if status != 0:
            raise PdhQueryError(
                "PdhGetFormattedCounterArrayW read failed with "
                f"{self._status_text(status)}"
            )
        items = ctypes.cast(
            raw, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM_W)
        )
        readings = []
        for index in range(item_count.value):
            item = items[index]
            readings.append(
                (
                    item.szName,
                    int(item.FmtValue.CStatus),
                    int(item.FmtValue.value.largeValue),
                )
            )
        return tuple(readings)

    def close_query(self, query) -> None:
        status = int(self._pdh.PdhCloseQuery(query))
        if status != 0:
            raise PdhQueryError(
                f"PdhCloseQuery failed with {self._status_text(status)}"
            )


# -- GPU sampler ---------------------------------------------------------


class GpuDedicatedMemorySampler:
    """Advisory GPU dedicated-memory sampler over PDH process counters.

    Matches only ``GPU Process Memory`` instances whose names begin with
    ``pid_<pid>_`` and sums ``Dedicated Usage`` across unique adapter/LUID
    instances.  Evidence from this sampler is advisory only and never gates
    any verdict; failures degrade to explicit availability states instead of
    fabricated zero readings.
    """

    __slots__ = ("_backend", "_closed", "_counter", "_pid", "_query")

    def __init__(self, pid: int, *, backend=None) -> None:
        if isinstance(pid, bool) or type(pid) is not int or pid < 1:
            raise ValueError("pid must be a positive integer")
        self._pid = pid
        self._backend = backend
        self._query = None
        self._counter = None
        self._closed = False

    @property
    def pid(self) -> int:
        return self._pid

    def open(self) -> GpuAvailability:
        """Bind the wildcard counter and publish only complete open state."""
        if self._closed:
            raise PdhQueryError("gpu sampler is closed")
        if self._query is not None or self._counter is not None:
            raise PdhQueryError("gpu sampler is already open")
        backend = self._backend
        if backend is None:
            backend = WindowsPdhBackend()
        query = backend.open_query()
        try:
            counter = backend.add_english_counter(
                query, GPU_PROCESS_MEMORY_COUNTER_PATH
            )
        except BaseException as primary:
            try:
                backend.close_query(query)
            except BaseException as close_failure:
                _add_bounded_secondary_note(
                    primary,
                    "gpu query rollback close also failed",
                    close_failure,
                )
            raise
        self._backend = backend
        self._query = query
        self._counter = counter
        return GpuAvailability.AVAILABLE

    def sample(self) -> GpuDedicatedSample:
        """Take one advisory reading of the matching PDH instances."""
        if self._closed or self._query is None or self._counter is None:
            return GpuDedicatedSample(GpuAvailability.UNSUPPORTED, None, 0)
        try:
            self._backend.collect(self._query)
            readings = self._backend.formatted_counter_array(self._counter)
        except PdhNoDataError:
            return GpuDedicatedSample(GpuAvailability.NO_ACTIVITY, None, 0)
        except Exception:
            return GpuDedicatedSample(GpuAvailability.UNRELIABLE, None, 0)

        prefix = f"pid_{self._pid}_"
        values_by_instance: dict[str, int] = {}
        malformed = False
        for name, counter_status, value in readings:
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            if (
                counter_status
                not in (PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA)
                or isinstance(value, bool)
                or type(value) is not int
                or value < 0
            ):
                malformed = True
                continue
            if name not in values_by_instance:
                values_by_instance[name] = value

        if malformed and not values_by_instance:
            return GpuDedicatedSample(GpuAvailability.UNRELIABLE, None, 0)
        if not values_by_instance:
            return GpuDedicatedSample(GpuAvailability.NO_ACTIVITY, None, 0)
        total = sum(values_by_instance.values())
        if total == 0:
            return GpuDedicatedSample(
                GpuAvailability.ZERO_OR_UNIFIED_MEMORY,
                0,
                len(values_by_instance),
            )
        return GpuDedicatedSample(
            GpuAvailability.AVAILABLE, total, len(values_by_instance)
        )

    def _release_query(self) -> None:
        query = self._query
        self._query = None
        self._counter = None
        if query is not None and self._backend is not None:
            # Clear ownership before the native call so a failed close is
            # never retried, but propagate its evidence to the monitor's
            # advisory cleanup diagnostics.
            self._backend.close_query(query)

    def close(self) -> None:
        """Close the PDH query deterministically; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._release_query()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
