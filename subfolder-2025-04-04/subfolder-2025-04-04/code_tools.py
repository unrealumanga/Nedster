"""Nedster Code Tools - Lint, test, format runners"""
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple


def detect_test_runner(cwd: str) -> str:
    """
    Check for: pytest.ini, pyproject.toml [tool.pytest],
    package.json scripts.test, Makefile test target.
    Return: "pytest" | "npm test" | "make test" | "unknown"
    """
    cwd_path = Path(cwd)

    # Check for pytest
    if (cwd_path / 'pytest.ini').exists():
        return 'pytest'

    if (cwd_path / 'pyproject.toml').exists():
        try:
            with open(cwd_path / 'pyproject.toml') as f:
                content = f.read()
                if '[tool.pytest' in content or 'pytest' in content:
                    return 'pytest'
        except Exception:
            pass

    if (cwd_path / 'setup.cfg').exists():
        try:
            with open(cwd_path / 'setup.cfg') as f:
                content = f.read()
                if '[tool:pytest]' in content or '[pytest]' in content:
                    return 'pytest'
        except Exception:
            pass

    # Check for npm test
    if (cwd_path / 'package.json').exists():
        try:
            import json
            with open(cwd_path / 'package.json') as f:
                pkg = json.load(f)
                if 'scripts' in pkg and 'test' in pkg['scripts']:
                    return 'npm test'
        except Exception:
            pass

    # Check for make test
    if (cwd_path / 'Makefile').exists() or (cwd_path / 'makefile').exists():
        try:
            with open(cwd_path / 'Makefile') as f:
                content = f.read()
                if 'test:' in content or '.PHONY: test' in content:
                    return 'make test'
        except Exception:
            pass

    return 'unknown'


def run_tests(cwd: str, file: str = "") -> str:
    """
    Run detected test suite. Capture output.
    Parse failures: extract "FAILED test_foo.py::test_bar" lines.
    Return: summary + failed test names
    """
    runner = detect_test_runner(cwd)

    if runner == 'unknown':
        return "No test runner detected (no pytest.ini, package.json, or Makefile found)"

    try:
        if runner == 'pytest':
            cmd = ['python', '-m', 'pytest']
            if file:
                cmd.append(file)
            else:
                cmd.extend(['-v', '--tb=short'])
        elif runner == 'npm test':
            cmd = ['npm', 'test']
            if file:
                cmd.append('--')
                cmd.append(file)
        elif runner == 'make test':
            cmd = ['make', 'test']
        else:
            return f"Unknown runner: {runner}"

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout + result.stderr

        # Parse failures
        failed_tests = []
        for line in output.split('\n'):
            if 'FAILED' in line:
                # Extract test name
                parts = line.split()
                for part in parts:
                    if '::' in part or part.endswith('.py'):
                        failed_tests.append(part)
            elif 'FAIL:' in line:
                failed_tests.append(line.split('FAIL:')[1].strip())

        # Summary
        passed = output.count(' PASSED')
        failed = len(failed_tests)
        errors = output.count(' ERROR')

        summary = f"Tests: {passed} passed, {failed} failed, {errors} errors\n"

        if failed_tests:
            summary += "\nFailed tests:\n" + "\n".join(f"  - {t}" for t in failed_tests[:10])

        if len(output) > 3000:
            summary += f"\n[Output truncated, {len(output)} chars total]"
        else:
            summary += f"\n{output}"

        return summary
    except subprocess.TimeoutExpired:
        return "Error: Tests timed out (60s limit)"
    except FileNotFoundError as e:
        return f"Error: Command not found - {e}"
    except Exception as e:
        return f"Error running tests: {e}"


def run_linter(cwd: str, file: str = "") -> str:
    """
    Check for: ruff, flake8, eslint, mypy.
    Run on file or whole project. Return issues summary.
    """
    cwd_path = Path(cwd)
    linters_found = []
    results = []

    # Check for ruff
    if check_command_exists('ruff'):
        linters_found.append('ruff')
        try:
            cmd = ['ruff', 'check']
            if file:
                cmd.append(file)

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout or result.returncode != 0:
                results.append(f"=== Ruff ===\n{result.stdout or result.stderr}")
        except Exception as e:
            results.append(f"Ruff error: {e}")

    # Check for flake8
    if check_command_exists('flake8'):
        linters_found.append('flake8')
        try:
            cmd = ['flake8']
            if file:
                cmd.append(file)

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout or result.returncode != 0:
                results.append(f"=== Flake8 ===\n{result.stdout or result.stderr}")
        except Exception as e:
            results.append(f"Flake8 error: {e}")

    # Check for eslint
    if (cwd_path / 'node_modules' / '.bin' / 'eslint').exists() or check_command_exists('eslint'):
        linters_found.append('eslint')
        try:
            eslint_path = cwd_path / 'node_modules' / '.bin' / 'eslint'
            if eslint_path.exists():
                cmd = [str(eslint_path)]
            else:
                cmd = ['eslint']

            if file:
                cmd.append(file)

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout or result.returncode != 0:
                results.append(f"=== ESLint ===\n{result.stdout or result.stderr}")
        except Exception as e:
            results.append(f"ESLint error: {e}")

    # Check for mypy
    if check_command_exists('mypy'):
        linters_found.append('mypy')
        try:
            cmd = ['mypy']
            if file:
                cmd.append(file)

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout or result.returncode != 0:
                results.append(f"=== MyPy ===\n{result.stdout or result.stderr}")
        except Exception as e:
            results.append(f"MyPy error: {e}")

    if not linters_found:
        return "No linters detected (ruff, flake8, eslint, mypy not found)"

    if not results:
        return f"Linters OK ({', '.join(linters_found)} found no issues)"

    return "\n\n".join(results)


def run_formatter(cwd: str, file: str = "") -> str:
    """
    Check for: black, ruff format, prettier.
    Run and return: "Formatted X files"
    """
    cwd_path = Path(cwd)
    results = []

    # Check for black
    if check_command_exists('black'):
        try:
            cmd = ['black', '--check'] if not file else ['black', '--check', file]
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # If check fails, run format
            if result.returncode != 0:
                cmd = ['black']
                if file:
                    cmd.append(file)

                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if 'reformatted' in result.stderr.lower():
                    count = result.stderr.count('reformatted')
                    results.append(f"Black: reformatted {count} file(s)")
                else:
                    results.append("Black: formatted 1 file")
            else:
                results.append("Black: no changes needed")
        except Exception as e:
            results.append(f"Black error: {e}")

    # Check for ruff format
    if check_command_exists('ruff'):
        try:
            cmd = ['ruff', 'format', '--check']
            if file:
                cmd.append(file)

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                cmd = ['ruff', 'format']
                if file:
                    cmd.append(file)

                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                results.append("Ruff: formatted files")
        except Exception as e:
            results.append(f"Ruff format error: {e}")

    # Check for prettier
    prettier_path = cwd_path / 'node_modules' / '.bin' / 'prettier'
    if prettier_path.exists() or check_command_exists('prettier'):
        try:
            cmd = [str(prettier_path) if prettier_path.exists() else 'prettier', '--check']
            if file:
                cmd.append(file)
            else:
                cmd.append('.')

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                cmd = [str(prettier_path) if prettier_path.exists() else 'prettier', '--write']
                if file:
                    cmd.append(file)
                else:
                    cmd.append('.')

                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if 'formatted' in result.stdout.lower():
                    results.append("Prettier: formatted files")
                else:
                    results.append("Prettier: formatted files")
        except Exception as e:
            results.append(f"Prettier error: {e}")

    if not results:
        return "No formatters detected (black, ruff, prettier not found)"

    return "\n".join(results)


def check_syntax(code: str, language: str = "python") -> str:
    """
    python: compile(code, '<string>', 'exec') - catch SyntaxError
    Return: "OK" or "SyntaxError line N: {msg}"
    """
    if language == "python":
        try:
            compile(code, '<string>', 'exec')
            return "OK"
        except SyntaxError as e:
            line_num = e.lineno or "unknown"
            msg = e.msg or "invalid syntax"
            return f"SyntaxError line {line_num}: {msg}"
    elif language == "json":
        import json
        try:
            json.loads(code)
            return "OK"
        except json.JSONDecodeError as e:
            return f"JSONError line {e.lineno}: {e.msg}"
    elif language == "javascript":
        # Try node if available
        try:
            result = subprocess.run(
                ['node', '--check'],
                input=code,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return "OK"
            else:
                return f"SyntaxError: {result.stderr.strip()[:200]}"
        except FileNotFoundError:
            return "OK (node not available for syntax check)"
        except Exception as e:
            return f"SyntaxError: {str(e)[:200]}"
    else:
        return f"OK (no syntax checker for {language})"


def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    import shutil
    return shutil.which(cmd) is not None


def get_project_info(cwd: str) -> dict:
    """
    Detect project structure and return info.
    Returns dict with: language, entry_point, test_runner, linters, formatters
    """
    cwd_path = Path(cwd)
    info = {
        'language': 'unknown',
        'entry_point': 'unknown',
        'test_runner': 'unknown',
        'linters': [],
        'formatters': [],
    }

    # Detect language
    py_files = list(cwd_path.glob('**/*.py'))
    js_files = list(cwd_path.glob('**/*.js'))
    ts_files = list(cwd_path.glob('**/*.ts'))
    go_files = list(cwd_path.glob('**/*.go'))

    if py_files:
        info['language'] = 'python'
    elif js_files:
        info['language'] = 'javascript'
    elif ts_files:
        info['language'] = 'typescript'
    elif go_files:
        info['language'] = 'go'

    # Detect entry point
    for entry in ['main.py', 'index.js', 'app.py', 'main.go', 'lib.rs']:
        if (cwd_path / entry).exists():
            info['entry_point'] = entry
            break

    # Detect test runner
    info['test_runner'] = detect_test_runner(cwd)

    # Detect linters
    if check_command_exists('ruff'):
        info['linters'].append('ruff')
    if check_command_exists('flake8'):
        info['linters'].append('flake8')
    if check_command_exists('eslint'):
        info['linters'].append('eslint')
    if check_command_exists('mypy'):
        info['linters'].append('mypy')

    # Detect formatters
    if check_command_exists('black'):
        info['formatters'].append('black')
    if check_command_exists('ruff'):
        info['formatters'].append('ruff format')
    if check_command_exists('prettier'):
        info['formatters'].append('prettier')

    return info
