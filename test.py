#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch evaluator for C++ submissions, following NOI Linux 2.0 judging rules."""

from __future__ import annotations

import argparse
import ctypes
import os
import shlex
import shutil
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
except ImportError:  # psutil is optional; memory monitoring is skipped without it.
    psutil = None


def _setup_windows_console() -> None:
    """Force UTF-8 console output and disable crash dialogs on Windows."""
    if sys.platform != "win32":
        return
    ctypes.windll.kernel32.SetConsoleCP(65001)
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]
    
    # Disable Windows Error Reporting (WER) dialogs for child processes.
    # This prevents the system from hanging while waiting for user input when a
    # submission segfaults, allowing the evaluator to catch the RE immediately.
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)


_setup_windows_console()


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
RUN_POLL_INTERVAL_SECONDS = 0.02


def _default_static_flag() -> str:
    """Return the static-link flag appropriate for the current platform.
    
    macOS and Windows typically don't need or support -static for local
    compilation in the same way Linux does, so we default to empty there.
    """
    return NOI_STATIC_FLAG if sys.platform.startswith("linux") else ""


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


def get_uss_mb(pid: int) -> Optional[float]:
    """Return the USS (unique set size) of a process in MiB, or None.

    USS counts memory owned exclusively by the process, which is much closer
    to the NOI memory accounting than RSS (which also includes shared libs).
    """
    if psutil is None:
        return None
    try:
        return psutil.Process(pid).memory_full_info().uss / 1024 / 1024
    except Exception:
        return None


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
    try:
        # Copy the input file into the work dir when the problem expects
        # file-based I/O.
        if input_name != "-":
            shutil.copy2(in_file, os.path.join(work_dir, input_name))

        # Remove any stale output file produced by a previous run.
        expected_output = out_file
        if output_name != "-":
            expected_output = os.path.join(work_dir, output_name)
            if os.path.exists(expected_output):
                os.remove(expected_output)

        # Crucial: Open files in BINARY mode ("rb" and "wb").
        # On Windows, text mode ("r" and "w") automatically translates "\r\n"
        # to "\n" and vice versa. This corrupts the byte stream compared to
        # the Linux judge environment.
        with open(in_file, "rb") as fin, \
             open(out_file, "wb") as fout:
            start = time.time()
            # Do NOT use stderr=PIPE here: a program writing lots of debug
            # output to stderr would fill the pipe buffer and block forever,
            # getting misjudged as TLE. DEVNULL discards it safely.
            proc = subprocess.Popen(
                [os.path.abspath(exe_path)],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.DEVNULL,
                cwd=work_dir,
            )

            # Poll for completion while enforcing TLE and MLE.
            while proc.poll() is None:
                elapsed = time.time() - start
                if elapsed > tle:
                    proc.kill()
                    proc.wait()
                    return "TLE", f"{elapsed:.2f}s"

                uss_mb = get_uss_mb(proc.pid)
                if uss_mb is not None and uss_mb > mle:
                    proc.kill()
                    proc.wait()
                    return "MLE", f"{uss_mb:.1f}MB"

                time.sleep(RUN_POLL_INTERVAL_SECONDS)

            return_code = proc.returncode
            if return_code != 0:
                # A negative exit code means the process was killed by a
                # signal (e.g. SIGSEGV), which is still RE.
                if return_code < 0:
                    return "RE", f"signal={-return_code}"
                return "RE", f"exit={return_code}"
            return "OK", f"{time.time() - start:.2f}s"
    except Exception as exc:
        if proc is not None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        return "ERR", str(exc)[:200]


def compare_files(student_output: str, expected_output: str) -> bool:
    """Compare two output files using the strict NOI filtering rule.
    
    Rules implemented:
    1. Ignore trailing spaces on each line.
    2. Ignore trailing newlines at the end of the file.
    3. Leading whitespace and interior blank lines are strictly compared.
    4. CRLF and LF differences are normalized to prevent Windows-specific WA.
    """
    def read_normalized(path: str) -> list[bytes]:
        with open(path, "rb") as f:
            content = f.read()
        
        # Normalize all line endings to \n. This removes Windows CRLF
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


def cleanup_problem_artifacts(
    prob_dir: str,
    data_dir: str,
    problem_name: str,
    input_name: str,
    output_name: str,
) -> None:
    """Remove artifacts generated during evaluation for a single problem."""
    filenames: set[str] = {f"{problem_name}.exe"}
    if input_name != "-":
        filenames.add(input_name)
    if output_name != "-":
        filenames.add(output_name)

    try:
        data_files = os.listdir(data_dir)
    except OSError:
        data_files = []

    # Also remove any testcase I/O files copied into the problem directory.
    for filename in data_files:
        stem, extension = os.path.splitext(filename)
        if filename.startswith(problem_name) and extension == ".in" and stem[len(problem_name):].isdigit():
            filenames.add(filename)
            filenames.add(stem + ".out")

    for filename in filenames:
        if filename == "-" or os.path.basename(filename) != filename:
            continue
        target = os.path.join(prob_dir, filename)
        try:
            os.remove(target)
        except FileNotFoundError:
            pass
        except OSError as error:
            print(f"Cleanup failed {target}: {error}", file=sys.stderr)


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
        if os.path.isdir(prob_dir):
            cleanup_problem_artifacts(prob_dir, data_dir, p_name, input_name, output_name)

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

        # Compile the submission with the configured standard and flags.
        exe_path = os.path.join(prob_dir, p_name + ".exe")
        ok, _ = compile_cpp(cpp_path, exe_path, cxx_standard, opt_flags, static_flag)
        if not ok:
            cleanup_problem_artifacts(prob_dir, data_dir, p_name, input_name, output_name)
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
            cleanup_problem_artifacts(prob_dir, data_dir, p_name, input_name, output_name)
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
        cleanup_problem_artifacts(prob_dir, data_dir, p_name, input_name, output_name)

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
        default=_default_static_flag(),
        help=(
            "Static-link flag. Pass an empty string to disable. "
            f"Default on this platform: {_default_static_flag()!r}."
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