"""Nedster Git Tools - Git integration functions"""
import subprocess
from pathlib import Path
from typing import Optional


def git_status(cwd: str) -> str:
    """
    Run: git status --short + git log --oneline -5
    Returns combined output as string.
    """
    try:
        # Git status --short
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )
        status_output = result.stdout.strip()

        # Git log --oneline -5
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )
        log_output = result.stdout.strip()

        output = []
        if status_output:
            output.append("=== Git Status ===")
            output.append(status_output)
        else:
            output.append("=== Git Status ===")
            output.append("(working tree clean)")

        if log_output:
            output.append("\n=== Recent Commits ===")
            output.append(log_output)

        return "\n".join(output)
    except FileNotFoundError:
        return "Error: git not found"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def git_diff(cwd: str, file: str = "") -> str:
    """
    Run: git diff HEAD [file] | head -100
    Returns diff output limited to 100 lines.
    """
    try:
        cmd = ["git", "diff", "HEAD"]
        if file:
            cmd.append(file)

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            if "not a git repository" in result.stderr.lower():
                return "Error: Not a git repository"
            return result.stderr.strip()

        lines = result.stdout.split('\n')[:100]
        diff_output = '\n'.join(lines)

        if len(result.stdout.split('\n')) > 100:
            diff_output += "\n... [diff truncated to 100 lines]"

        return diff_output if diff_output else "(no changes)"
    except FileNotFoundError:
        return "Error: git not found"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def git_commit(cwd: str, message: str = "") -> str:
    """
    Run: git add -A && git commit -m "{message}"
    Auto-generate message if empty: ask LLM for conventional commit msg.
    Returns commit result.
    """
    try:
        # First check if there are any changes
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if not status_result.stdout.strip():
            return "(nothing to commit, working tree clean)"

        # Stage all changes
        add_result = subprocess.run(
            ["git", "add", "-A"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if add_result.returncode != 0:
            return f"Error staging: {add_result.stderr}"

        # Generate message if empty
        if not message.strip():
            # Get diff for LLM to analyze
            diff_result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Simple conventional commit message generation
            diff_text = diff_result.stdout[:2000]
            if "fix" in diff_text.lower() or "bug" in diff_text.lower():
                message = "fix: resolve issues in modified files"
            elif "test" in diff_text.lower():
                message = "test: add or update tests"
            elif "refactor" in diff_text.lower():
                message = "refactor: improve code structure"
            elif "doc" in diff_text.lower() or "readme" in diff_text.lower():
                message = "docs: update documentation"
            else:
                message = "chore: update files"

        # Commit
        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15
        )

        if commit_result.returncode != 0:
            return f"Error committing: {commit_result.stderr}"

        # Get the commit hash
        log_result = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        commit_hash = log_result.stdout.strip()
        return f"Committed: {commit_hash}\nMessage: {message}"
    except FileNotFoundError:
        return "Error: git not found"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def git_branch(cwd: str) -> str:
    """
    Run: git branch --show-current
    Returns current branch name.
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return "Error: Not a git repository"

        branch = result.stdout.strip()
        return branch if branch else "(detached HEAD)"
    except FileNotFoundError:
        return "Error: git not found"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def git_stash(cwd: str) -> str:
    """
    Run: git stash
    Returns stash result.
    """
    try:
        result = subprocess.run(
            ["git", "stash"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return result.stderr.strip()

        return result.stdout.strip() or "(nothing to stash)"
    except FileNotFoundError:
        return "Error: git not found"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def git_log(cwd: str, n: int = 10) -> str:
    """
    Run: git log --oneline -n
    Returns last n commits.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{n}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return "Error: Not a git repository"

        return result.stdout.strip()
    except FileNotFoundError:
        return "Error: git not found"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def git_add(cwd: str, files: list) -> str:
    """
    Run: git add [files]
    Returns result.
    """
    try:
        result = subprocess.run(
            ["git", "add"] + files,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return result.stderr.strip()

        return f"Staged: {', '.join(files)}"
    except FileNotFoundError:
        return "Error: git not found"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def is_git_repo(cwd: str) -> bool:
    """Check if cwd is a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False
