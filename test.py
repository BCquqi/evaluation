#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch evaluator for C++ submissions, following NOI Linux 2.0 judging rules."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from typing import NamedTuple, Optional
from xml.sax.saxutils import escape

try:
    import psutil  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports,reportMissingModuleSource]
except ImportError:  # psutil is optional; it only backs the platform fallbacks.
    psutil = None

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    import resource


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROBLEM_COUNT = 4
PROBLEM_POINTS = 100

# Higher value = higher priority when aggregating per-testcase statuses.
STATUS_PRIORITY = {
    "WA": 1,
    "NO_ANS": 1,
    "RE": 2,
    "ERR": 2,
    "TLE": 3,
    "MLE": 4,
}

# NOI Linux 2.0 official compile options. The CLI can override each of these.
NOI_CXX_STANDARD = "c++14"
NOI_OPT_FLAGS = "-O2"
NOI_STATIC_FLAG = "-static"

COMPILE_TIMEOUT_SECONDS = 30
RUN_POLL_INTERVAL_SECONDS = 0.005

# NOI judges measure CPU time (user+system) against the time limit and peak
# resident memory against the memory limit, both read after the process exits.
# Wall-clock is used only as an emergency brake, so that a blocked or sleeping
# submission cannot hang the whole evaluation run. It never produces a verdict
# on its own unless the CPU-time reading is unavailable.
WALL_CLOCK_FACTOR = 3.0
WALL_CLOCK_SLACK_SECONDS = 5.0

# A hard OS ceiling that stops a runaway submission from exhausting the host.
# This is deliberately far above the judged limit: when it fires the process
# merely fails to allocate, which is indistinguishable from a normal exit, so
# it can only ever be a safety net. The memory verdict comes from the kernel's
# recorded peak instead.
MEMORY_SAFETY_FACTOR = 4
MEMORY_SAFETY_MIN_MB = 1024
MEMORY_SAFETY_MAX_MB = 2048

# Windows Job Object layout. Queried after the run for PeakProcessMemoryUsed,
# the kernel's exact high-water mark for the process (not a sample).
if sys.platform == "win32":
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _SEM_FAILCRITICALERRORS = 0x0001
    _SEM_NOGPFAULTERRORBOX = 0x0002
    _CREATE_NO_WINDOW = 0x08000000

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # Without this a crashing submission blocks on a Windows Error Reporting
    # dialog and gets misread as TLE by the wall-clock brake.
    _kernel32.SetErrorMode(_SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX)

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class Usage(NamedTuple):
    """Measured resource usage of a single execution."""

    cpu_seconds: Optional[float]
    peak_memory_mb: Optional[float]


def _memory_ceiling_mb(mle: float) -> int:
    """Hard OS memory ceiling for one run, well above the judged limit."""
    wanted = max(mle * MEMORY_SAFETY_FACTOR, MEMORY_SAFETY_MIN_MB)
    return int(min(wanted, MEMORY_SAFETY_MAX_MB))


def _posix_exit_code(status: int) -> int:
    """Translate a raw wait status into Popen's exit-code convention."""
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    return -1



def _default_static_flag() -> str:
    """Return the static-link flag appropriate for the current platform.
    
    macOS typically doesn't support -static for local compilation, so we
    default to empty there.
    """
    return NOI_STATIC_FLAG if sys.platform != "darwin" else ""


class Problem(NamedTuple):
    """Resolved problem configuration used during evaluation."""

    name: str
    tle: float
    mle: float
    input_name: str
    output_name: str


def get_problems(
    problem_count: int,
    problem_configs: dict[str, tuple[str, str, str, float, float]],
) -> list[Problem]:
    """Resolve the raw CLI configuration into a list of Problem entries."""
    problems: list[Problem] = []
    for i in range(1, problem_count + 1):
        problem_id = str(i)
        config = problem_configs.get(problem_id)
        if config is None:
            raise ValueError(f"Missing configuration for problem {problem_id}")
        name, input_name, output_name, tle, mle = config
        problems.append(Problem(name, tle, mle, input_name, output_name))
    return problems


def get_testcase_points(testcase_count: int) -> list[int]:
    """Distribute PROBLEM_POINTS across testcases; extra points go to the tail."""
    base_points, remainder = divmod(PROBLEM_POINTS, testcase_count)
    points = [base_points] * testcase_count
    for index in range(testcase_count - 1, testcase_count - remainder - 1, -1):
        points[index] += 1
    return points


def compile_cpp(
    src_path: str,
    exe_path: str,
    cxx_standard: str = NOI_CXX_STANDARD,
    opt_flags: str = NOI_OPT_FLAGS,
    static_flag: str = NOI_STATIC_FLAG,
) -> tuple[bool, str]:
    """Compile a C++ source with the configured standard and flags.

    opt_flags is a shell-like string; shlex.split turns it into an argv list
    so multiple flags (e.g. "-O2 -Wall") can be supplied in one argument.
    An empty static_flag disables static linking.
    """
    command = ["g++"]
    if cxx_standard:
        command.append(f"-std={cxx_standard}")
    if opt_flags:
        try:
            command.extend(shlex.split(opt_flags))
        except ValueError as exc:
            return False, f"Invalid opt flags: {exc}"
    if static_flag:
        command.append(static_flag)
    command.extend(["-o", str(exe_path), str(src_path)])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except FileNotFoundError:
        return False, "g++ not found"

    if result.returncode == 0:
        return True, ""

    error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    return False, error[:200]


def _win32_cpu_seconds(handle: int) -> Optional[float]:
    """Read user+system CPU seconds from an open process handle."""
    created = wintypes.FILETIME()
    exited = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not _kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return None
    ticks = ((kernel.dwHighDateTime << 32 | kernel.dwLowDateTime)
             + (user.dwHighDateTime << 32 | user.dwLowDateTime))
    # FILETIME counts 100-nanosecond intervals; all threads are included.
    return ticks / 1e7


class Meter:
    """Collects exact CPU time and peak memory for one program execution.

    The handles stay open until after the child exits, because that is the only
    moment the kernel can report the final totals. On Windows the memory figure
    is the Job Object's PeakProcessMemoryUsed, i.e. the real high-water mark,
    which sampling can never reproduce for short allocation bursts.
    """

    def __init__(self, memory_ceiling_mb: int) -> None:
        self.error = ""
        self._proc_handle = 0
        self._job_handle = 0
        self._pid = 0
        self._last_sampled = 0.0
        self._posix_usage: Optional[Usage] = None
        if sys.platform == "win32":
            self._job_handle = _kernel32.CreateJobObjectW(None, None) or 0
            if not self._job_handle:
                self.error = "CreateJobObject failed"
                return
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            info.ProcessMemoryLimit = memory_ceiling_mb * 1024 * 1024
            if not _kernel32.SetInformationJobObject(
                self._job_handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                self.error = "SetInformationJobObject failed"

    def attach(self, pid: int) -> bool:
        """Bind the running child to this meter."""
        if self.error:
            return False
        if sys.platform != "win32":
            self._pid = pid
            return True
        # Popen returns before the child is guaranteed openable. The window is
        # microseconds wide, but a few retries keep very short programs from
        # being reported as evaluator errors.
        for _ in range(50):
            self._proc_handle = _kernel32.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE | _PROCESS_SET_QUOTA,
                False,
                pid,
            )
            if self._proc_handle:
                break
            time.sleep(0.001)
        if not self._proc_handle:
            self.error = "OpenProcess failed"
            return False
        if not _kernel32.AssignProcessToJobObject(self._job_handle, self._proc_handle):
            # A process already inside a non-nestable job cannot be reassigned.
            # Fall back to sampled peaks rather than failing the whole case.
            self.error = "AssignProcessToJobObject failed"
            return False
        return True

    def sample(self, pid: int) -> None:
        """Record a live memory sample, used only as a last-resort fallback."""
        if psutil is None:
            return
        try:
            process = psutil.Process(pid)
            peak = getattr(process.memory_info(), "peak_wset", 0)
            if peak:
                self._last_sampled = peak / 1024 / 1024
            else:
                self._last_sampled = process.memory_full_info().uss / 1024 / 1024
        except Exception:
            pass

    def terminate(self, pid: int) -> None:
        """Kill the child and anything it spawned (best effort)."""
        if sys.platform == "win32":
            if self._proc_handle:
                _kernel32.TerminateProcess(self._proc_handle, 1)
            return
        if not pid:
            return
        # The child was started in its own session, so its whole process group
        # can be signalled at once (a fork bomb or shell pipeline included).
        # pid 0 would mean this evaluator's own group, hence the guard above.
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    def poll(self, proc: subprocess.Popen) -> Optional[int]:
        """Return the exit code once the child has finished, else None.

        POSIX uses wait4 so that the child's own rusage is captured. Popen
        would otherwise only expose the superprocess-wide cumulative totals.
        """
        if sys.platform == "win32":
            return proc.poll()
        try:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        except ChildProcessError:
            return proc.returncode
        if pid == 0:
            return None
        self._posix_usage = Usage(usage.ru_utime + usage.ru_stime, usage.ru_maxrss / 1024)
        proc.returncode = _posix_exit_code(status)
        return proc.returncode

    def wait(self, proc: subprocess.Popen) -> int:
        """Block until the child exits and record its rusage."""
        if sys.platform == "win32":
            return proc.wait()
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        except ChildProcessError:
            return proc.returncode if proc.returncode is not None else -1
        self._posix_usage = Usage(usage.ru_utime + usage.ru_stime, usage.ru_maxrss / 1024)
        proc.returncode = _posix_exit_code(status)
        return proc.returncode

    def finish(self) -> Usage:
        """Close the handles and return the exact recorded usage."""
        if sys.platform != "win32":
            return self._posix_usage or Usage(None, self._last_sampled or None)

        cpu: Optional[float] = None
        peak: Optional[float] = None
        if self._proc_handle:
            cpu = _win32_cpu_seconds(self._proc_handle)
        if self._job_handle:
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            if _kernel32.QueryInformationJobObject(
                self._job_handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
                None,
            ):
                peak = info.PeakProcessMemoryUsed / 1024 / 1024
        self.close()

        if peak is None or peak <= 0.0:
            peak = self._last_sampled or None
        return Usage(cpu, peak)

    def close(self) -> None:
        if sys.platform == "win32":
            if self._proc_handle:
                _kernel32.CloseHandle(self._proc_handle)
                self._proc_handle = 0
            if self._job_handle:
                _kernel32.CloseHandle(self._job_handle)
                self._job_handle = 0


def run_program(
    exe_path: str,
    in_file: str,
    out_file: str,
    tle: float,
    mle: float,
    work_dir: str,
    input_name: str,
    output_name: str,
) -> tuple[str, str]:
    """Execute a compiled program against a single input file.

    Returns a (status, detail) tuple where status is one of:
        "OK"  - program finished normally
        "TLE" - time limit exceeded
        "MLE" - memory limit exceeded
        "RE"  - non-zero exit code or killed by a signal
        "ERR" - evaluator-side failure
    """
    proc: Optional[subprocess.Popen] = None
    ceiling_mb = _memory_ceiling_mb(mle)
    meter = Meter(ceiling_mb)
    if meter.error:
        return "ERR", meter.error
    try:
        # Copy the input file into the work dir when the problem expects
        # file-based I/O (e.g., freopen).
        if input_name != "-":
            shutil.copy2(in_file, os.path.join(work_dir, input_name))

        # Remove any stale output file produced by a previous run.
        if output_name != "-":
            expected_output = os.path.join(work_dir, output_name)
            if os.path.exists(expected_output):
                os.remove(expected_output)

        # Crucial: Open files in BINARY mode ("rb" and "wb").
        # Text mode ("r" and "w") automatically translates line endings on
        # some platforms, corrupting the byte stream compared to the Linux
        # judge environment.
        with open(in_file, "rb") as fin, open(out_file, "wb") as fout:
            start = time.time()
            # Do NOT use stderr=PIPE here: a program writing lots of debug
            # output to stderr would fill the pipe buffer and block forever,
            # getting misjudged as TLE. DEVNULL discards it safely.
            #
            # start_new_session (POSIX) and CREATE_NO_WINDOW (Windows) both
            # detach the child so the whole tree can be killed on a brake
            # without a console window flashing up.
            popen_kwargs = {}
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = _CREATE_NO_WINDOW
            else:
                popen_kwargs["start_new_session"] = True
                # Same safety net as the Windows job limit: cap the address
                # space well above the judged limit so a runaway allocation
                # cannot take the host down. A verdict never comes from this.
                ceiling_bytes = ceiling_mb * 1024 * 1024

                def _limit_address_space() -> None:
                    resource.setrlimit(
                        resource.RLIMIT_AS, (ceiling_bytes, ceiling_bytes)
                    )

                popen_kwargs["preexec_fn"] = _limit_address_space
            proc = subprocess.Popen(
                [os.path.abspath(exe_path)],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.DEVNULL,
                cwd=work_dir,
                **popen_kwargs,
            )
            meter.attach(proc.pid)

            # The loop is only an emergency brake: a program that sleeps or
            # blocks burns no CPU, so it would otherwise run forever. Verdicts
            # are decided below from the exact post-exit measurements.
            broken_by_brake = False
            wall_cap = max(tle * WALL_CLOCK_FACTOR, tle + WALL_CLOCK_SLACK_SECONDS)
            while meter.poll(proc) is None:
                elapsed = time.time() - start
                meter.sample(proc.pid)
                if elapsed > wall_cap:
                    broken_by_brake = True
                    meter.terminate(proc.pid)
                    meter.wait(proc)
                    break
                time.sleep(RUN_POLL_INTERVAL_SECONDS)

            usage = meter.finish()
            cpu = usage.cpu_seconds
            note = (
                f"{cpu:.2f}s cpu, {usage.peak_memory_mb:.1f}MB"
                if cpu is not None and usage.peak_memory_mb is not None
                else f"{time.time() - start:.2f}s wall"
            )

            # 1. Memory first: a program killed at the ceiling is already MLE
            #    even if it also burned a lot of time getting there.
            if usage.peak_memory_mb is not None and usage.peak_memory_mb > mle:
                return "MLE", note

            # 2. CPU time against the limit, exactly like NOI judges. Ties go
            #    to the judge: exceeding the limit in either direction is TLE.
            if cpu is not None and cpu > tle:
                return "TLE", note

            # 3. The brake fired but the program was under both limits — that
            #    only happens for a genuinely stalled process (deadlock,
            #    blocking read). Report TLE, since it would never finish.
            if broken_by_brake:
                return "TLE", f"{time.time() - start:.2f}s wall, no verdict"

            return_code = proc.returncode
            if return_code != 0:
                # A negative exit code means the process was killed by a
                # signal (e.g. SIGSEGV), which is still RE.
                if return_code < 0:
                    return "RE", f"signal={-return_code}"
                # Windows reports access violations as a positive NTSTATUS.
                if sys.platform == "win32" and return_code >= 0xC0000000:
                    return "RE", f"exception=0x{return_code:x}"
                return "RE", f"exit={return_code}"
            return "OK", note
    except Exception as exc:
        meter.terminate(proc.pid if proc is not None else 0)
        if proc is not None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        meter.close()
        return "ERR", str(exc)[:200]


def compare_files(student_output: str, expected_output: str) -> bool:
    """Compare two output files using the strict NOI filtering rule.
    
    Rules implemented:
    1. Ignore trailing spaces on each line.
    2. Ignore trailing newlines at the end of the file.
    3. Leading whitespace and interior blank lines are strictly compared.
    4. CRLF and LF differences are normalized to prevent cross-platform WA.
    """
    def read_normalized(path: str) -> list[bytes]:
        with open(path, "rb") as f:
            content = f.read()
        
        # Normalize all line endings to \n. This removes CRLF
        # differences and ensures cross-platform consistency.
        content = content.replace(b"\r\n", b"\n")
        
        lines = content.split(b"\n")
        
        # Strip trailing spaces from each line (NOI rule).
        # Leading spaces are preserved to catch formatting errors.
        lines = [line.rstrip(b" ") for line in lines]
        
        # Drop trailing empty lines (NOI rule: ignore file-ending newlines).
        while lines and not lines[-1]:
            lines.pop()
            
        return lines

    try:
        return read_normalized(student_output) == read_normalized(expected_output)
    except Exception:
        return False


def exe_file_name(problem_name: str) -> str:
    """Return the executable name for a problem on this platform.

    g++ on Windows appends .exe to the -o path regardless of what was asked
    for, so the reported name must carry it; otherwise cleanup misses the file
    and a stale binary can survive into the next run.
    """
    return problem_name + ".exe" if sys.platform == "win32" else problem_name


def cleanup_problem_artifacts(prob_dir: str, problem_name: str) -> None:
    """Remove the executable generated during evaluation."""
    exe_path = os.path.join(prob_dir, exe_file_name(problem_name))
    try:
        os.remove(exe_path)
    except FileNotFoundError:
        pass
    except OSError as error:
        print(f"Cleanup failed {exe_path}: {error}", file=sys.stderr)


def run_testcase(
    exe_path: str,
    src_in: str,
    src_ans: str,
    in_file: str,
    tle: float,
    mle: float,
    input_name: str,
    output_name: str,
) -> str:
    """Run one testcase in an isolated temp dir and return its verdict."""
    try:
        # Use an isolated temporary directory for each testcase so that
        # submissions using freopen() won't overwrite each other.
        with tempfile.TemporaryDirectory(prefix="oj_case_") as work_dir:
            out_file = os.path.join(work_dir, in_file.replace(".in", ".out"))
            status, _ = run_program(
                exe_path, src_in, out_file, tle, mle, work_dir, input_name, output_name,
            )
            if status != "OK":
                return status
            student_output = out_file if output_name == "-" else os.path.join(work_dir, output_name)
            return "AC" if compare_files(student_output, src_ans) else "WA"
    except Exception:
        return "ERR"


def evaluate_student(
    student_id: int,
    prefix: str,
    width: int,
    data_dir: str,
    problems: list[Problem],
    quiet: bool = False,
    parent_dir: str = PARENT_DIR,
    cxx_standard: str = NOI_CXX_STANDARD,
    opt_flags: str = NOI_OPT_FLAGS,
    static_flag: str = NOI_STATIC_FLAG,
) -> Optional[dict]:
    """Evaluate all problems for one student and return a per-problem result."""
    folder_name = f"{prefix}{student_id:0{width}d}"
    folder_path = os.path.join(parent_dir, folder_name)

    if not os.path.isdir(folder_path):
        if not quiet:
            print(f"Skip {folder_name}: missing")
        return None

    if not quiet:
        print(f"\n=== {folder_name} ===")

    student_result: dict = {"folder": folder_name, "problems": {}}

    def set_problem_result(problem_name: str, status: str, score: int = 0) -> None:
        student_result["problems"][problem_name] = {"status": status, "score": score}

    for problem in problems:
        p_name, tle, mle, input_name, output_name = problem
        prob_dir = os.path.join(folder_path, p_name)
        
        # Clean up previous run's executable
        cleanup_problem_artifacts(prob_dir, p_name)

        cpp_path = os.path.join(prob_dir, p_name + ".cpp")

        if not os.path.isdir(prob_dir):
            set_problem_result(p_name, "NO_DIR")
            if not quiet:
                print(f"  {p_name}: NO_DIR")
            continue

        if not os.path.isfile(cpp_path):
            set_problem_result(p_name, "NO_CPP")
            if not quiet:
                print(f"  {p_name}: NO_CPP")
            continue

        # Output executable name for this platform (.exe on Windows).
        exe_path = os.path.join(prob_dir, exe_file_name(p_name))
        ok, _ = compile_cpp(cpp_path, exe_path, cxx_standard, opt_flags, static_flag)
        if not ok:
            cleanup_problem_artifacts(prob_dir, p_name)
            set_problem_result(p_name, "CE")
            if not quiet:
                print(f"  {p_name}: CE")
            continue

        # Discover testcases named <problem><case>.in.
        in_files = sorted(
            (
                f for f in os.listdir(data_dir)
                if f.startswith(p_name)
                and f.endswith(".in")
                and os.path.splitext(f)[0][len(p_name):].isdigit()
            ),
            key=lambda f: int(os.path.splitext(f)[0][len(p_name):]),
        )

        if not in_files:
            cleanup_problem_artifacts(prob_dir, p_name)
            set_problem_result(p_name, "NO_TESTCASES")
            if not quiet:
                print(f"  {p_name}: NO_TESTCASES")
            continue

        testcase_statuses: list[str] = []
        score = 0
        testcase_points = get_testcase_points(len(in_files))

        for i, (in_file, testcase_score) in enumerate(zip(in_files, testcase_points), 1):
            src_in = os.path.join(data_dir, in_file)

            # Prefer .ans over .out when both exist.
            src_ans = next(
                (
                    os.path.join(data_dir, candidate)
                    for candidate in (
                        in_file.replace(".in", ".ans"),
                        in_file.replace(".in", ".out"),
                    )
                    if os.path.isfile(os.path.join(data_dir, candidate))
                ),
                None,
            )

            if src_ans is None:
                testcase_statuses.append("NO_ANS")
                if not quiet:
                    print(f"  {p_name}{i}: NO_ANS")
                continue

            status = run_testcase(
                exe_path, src_in, src_ans, in_file, tle, mle, input_name, output_name,
            )
            testcase_statuses.append(status)
            if status == "AC":
                score += testcase_score
            elif not quiet:
                print(f"  {p_name}{i}: {status}")

        # Aggregate per-testcase verdicts into one problem verdict.
        if all(status == "AC" for status in testcase_statuses):
            problem_status = "AC"
        else:
            problem_status = max(
                (status for status in testcase_statuses if status != "AC"),
                key=lambda status: STATUS_PRIORITY.get(status, 0),
                default="ERROR",
            )

        set_problem_result(p_name, problem_status, score)
        cleanup_problem_artifacts(prob_dir, p_name)

        if not quiet:
            print(f"  {p_name}: {problem_status} {score}/{PROBLEM_POINTS}")

    return student_result


def print_summary(results: list[Optional[dict]], problems: list[Problem]) -> None:
    """Print the aggregate 'TEST <n>, AK <m>' summary line."""
    evaluated = [r for r in results if r is not None]
    all_correct = sum(
        all(r["problems"].get(p.name, {}).get("status") == "AC" for p in problems)
        for r in evaluated
    )
    print(f"TEST {len(evaluated)}, AK {all_correct}")


def excel_column_name(index: int) -> str:
    """Convert a 1-based column index to its Excel letter (1 -> A, 27 -> AA)."""
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def excel_cell_xml(row: int, column: int, value) -> str:
    """Serialize a single cell as inline string or numeric SpreadsheetML."""
    reference = f"{excel_column_name(column)}{row}"
    if isinstance(value, (int, float)):
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_excel_report(
    results: list[Optional[dict]],
    problems: list[Problem],
    output_dir: str,
    cxx_standard: str = NOI_CXX_STANDARD,
    opt_flags: str = NOI_OPT_FLAGS,
    static_flag: str = NOI_STATIC_FLAG,
) -> str:
    """Write a minimal .xlsx report to output_dir and return its path."""
    headers = ["id"]
    for problem in problems:
        headers.extend([problem.name, problem.name])
    headers.append("total")

    # Build the config header from the actual compile settings.
    compile_parts = ["g++"]
    if cxx_standard:
        compile_parts.append(f"-std={cxx_standard}")
    if opt_flags:
        compile_parts.append(opt_flags)
    if static_flag:
        compile_parts.append(static_flag)
    configs = [" ".join(compile_parts)]
    for problem in problems:
        configs.append(
            f"{problem.name} {problem.input_name} {problem.output_name} "
            f"{problem.tle}s {problem.mle}MB {PROBLEM_POINTS}"
        )
    rows: list[list] = [[" | ".join(configs)], headers]

    for result in results:
        if result is None:
            continue
        row: list = [result["folder"]]
        total_score = 0
        for problem in problems:
            problem_result = result["problems"].get(problem.name, {"status": "?", "score": 0})
            score = problem_result["score"]
            total_score += score
            row.extend([score, problem_result["status"]])
        row.append(total_score)
        rows.append(row)

    last_column = excel_column_name(len(headers))
    xml_rows = []
    for row_index, values in enumerate(rows, 1):
        cells = "".join(
            excel_cell_xml(row_index, column_index, value)
            for column_index, value in enumerate(values, 1)
        )
        xml_rows.append(f'<row r="{row_index}">{cells}</row>')

    column_widths = ['<col min="1" max="1" width="18" customWidth="1"/>']
    column_widths.extend(
        f'<col min="{column}" max="{column}" width="12" customWidth="1"/>'
        for column in range(2, len(headers) + 1)
    )

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_column}{len(rows)}"/>'
        f'<cols>{"".join(column_widths)}</cols>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f'<mergeCells count="1"><mergeCell ref="A1:{last_column}1"/></mergeCells>'
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Results" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    package_relationships_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_relationships_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    report_path = os.path.join(
        output_dir,
        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    )
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(report_path, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml)
        workbook.writestr("_rels/.rels", package_relationships_xml)
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_relationships_xml)
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and resolve the problem configuration."""
    parser = argparse.ArgumentParser(description="Batch C++ evaluator (NOI Linux 2.0)")

    parser.add_argument("--prefix", default="SH-J", help="Student folder prefix")
    parser.add_argument("--start", type=int, default=1, help="Start student ID")
    parser.add_argument("--end", type=int, default=5, help="End student ID")
    parser.add_argument("--width", type=int, default=5, help="Zero-padding width for student ID")
    parser.add_argument("--problem-count", type=int, default=DEFAULT_PROBLEM_COUNT, help="Number of problems")

    parser.add_argument("--code-dir", default=PARENT_DIR, help="Parent directory of student code folders")
    parser.add_argument("--data-dir", default=os.path.join(PARENT_DIR, "data"), help="Test data directory")
    parser.add_argument("--output-dir", help="Directory for result files")

    parser.add_argument(
        "--problem-config",
        action="append",
        nargs=6,
        metavar=("ID", "NAME", "INPUT_NAME", "OUTPUT_NAME", "TIME_LIMIT", "MEMORY_MB"),
        help="Configure a problem; use - for standard input/output. Repeat per problem.",
    )

    # Compile settings. Each defaults to the NOI Linux 2.0 value, so leaving
    # them untouched reproduces the original behavior.
    default_static = _default_static_flag()

    parser.add_argument(
        "--cxx-standard",
        default=NOI_CXX_STANDARD,
        help=f"C++ standard, e.g. c++14 (default: {NOI_CXX_STANDARD}).",
    )
    parser.add_argument(
        "--opt-flags",
        default=NOI_OPT_FLAGS,
        help=(
            "Optimization flags passed to g++; quote the value when it contains "
            f"spaces. Default: {NOI_OPT_FLAGS}."
        ),
    )
    parser.add_argument(
        "--static-flag",
        default=default_static,
        help=(
            "Static-link flag. Pass an empty string to disable. "
            f"Default on this platform: {default_static!r}."
        ),
    )

    args = parser.parse_args()

    args.code_dir = os.path.abspath(args.code_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir or args.code_dir)

    if args.problem_count < 1:
        parser.error("Problem count must be at least 1")

    problem_configs: dict[str, tuple[str, str, str, float, float]] = {}
    for raw_config in args.problem_config or []:
        problem_id, name, input_name, output_name, time_limit, memory_mb = raw_config
        if not problem_id.isdigit() or not 1 <= int(problem_id) <= args.problem_count:
            parser.error(f"Problem ID must be between 1 and {args.problem_count}: {problem_id}")
        if problem_id in problem_configs:
            parser.error(f"Duplicate problem ID: {problem_id}")
        if not name or os.path.basename(name) != name or name in (".", ".."):
            parser.error(f"Problem name must be a directory name: {name}")
        if input_name != "-" and output_name != "-" and input_name == output_name:
            parser.error(f"Problem {problem_id}: input and output filenames must differ")
        try:
            time_limit = float(time_limit)
            memory_mb = float(memory_mb)
        except ValueError:
            parser.error("Time and memory limits must be numbers")
        if time_limit <= 0 or memory_mb <= 0:
            parser.error("Time and memory limits must be positive")
        problem_configs[problem_id] = (name, input_name, output_name, time_limit, memory_mb)

    missing_ids = [str(i) for i in range(1, args.problem_count + 1) if str(i) not in problem_configs]
    if missing_ids:
        parser.error("Missing --problem-config for: " + ", ".join(missing_ids))

    args.problems = get_problems(args.problem_count, problem_configs)
    return args


def main() -> None:
    args = parse_args()
    print(f"{args.prefix} {args.start}-{args.end} {args.width} {args.data_dir} {args.problem_count}")
    for problem in args.problems:
        print(
            f"{problem.name} {problem.input_name} {problem.output_name} "
            f"{problem.tle}s {problem.mle}MB"
        )

    results = []
    for student_id in range(args.start, args.end + 1):
        result = evaluate_student(
            student_id,
            args.prefix,
            args.width,
            args.data_dir,
            args.problems,
            quiet=True,
            parent_dir=args.code_dir,
            cxx_standard=args.cxx_standard,
            opt_flags=args.opt_flags,
            static_flag=args.static_flag,
        )
        results.append(result)

    print_summary(results, args.problems)
    report_path = write_excel_report(
        results,
        args.problems,
        args.output_dir,
        cxx_standard=args.cxx_standard,
        opt_flags=args.opt_flags,
        static_flag=args.static_flag,
    )
    print(report_path)


if __name__ == "__main__":
    main()