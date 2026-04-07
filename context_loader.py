"""Nedster ContextLoader - Project context builder with smart file selection"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pathspec
import tiktoken


class ContextLoader:
    def __init__(self, project_dir: str):
        self.root = Path(project_dir)
        self.file_index: Dict[str, Tuple[int, float, str]] = {}  # path -> (size, mtime, language)
        self.active_files: List[str] = []
        self.gitignore_spec: Optional[pathspec.PathSpec] = None
        self._skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.gitignore'}
        self._skip_exts = {'.pyc', '.so', '.bin', '.png', '.jpg', '.jpeg', '.gif', '.lock'}

    def scan_project(self) -> int:
        """
        Walk project dir. Skip: .git, __pycache__, node_modules, .venv, venv, *.pyc, *.so, *.bin, images, lock files.
        Use pathspec to honor .gitignore.
        Build self.file_index.
        Returns count of files scanned.
        """
        self.file_index = {}

        # Load .gitignore if present
        gitignore_path = self.root / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path) as f:
                    self.gitignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', f)
            except Exception:
                self.gitignore_spec = None

        file_count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Skip ignored directories
            dirnames[:] = [d for d in dirnames if d not in self._skip_dirs]

            rel_dir = Path(dirpath).relative_to(self.root)

            for filename in filenames:
                # Skip ignored extensions
                if any(filename.endswith(ext) for ext in self._skip_exts):
                    continue

                filepath = Path(dirpath) / filename
                rel_path = str(filepath.relative_to(self.root))

                # Check gitignore
                if self.gitignore_spec and self.gitignore_spec.match_file(rel_path):
                    continue

                # Get file info
                try:
                    stat = filepath.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime
                    language = self._detect_language(filename)

                    if size > 0 and size < 500000:  # Skip files > 500KB
                        self.file_index[rel_path] = (size, mtime, language)
                        file_count += 1
                except Exception:
                    continue

        print(f"[Nedster] Scanned {file_count} files in {self.root.name}")
        return file_count

    def _detect_language(self, filename: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'javascript', '.tsx': 'typescript', '.java': 'java',
            '.go': 'go', '.rs': 'rust', '.c': 'c', '.cpp': 'cpp',
            '.h': 'c', '.hpp': 'cpp', '.rb': 'ruby', '.php': 'php',
            '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
            '.sh': 'shell', '.bash': 'shell', '.zsh': 'shell',
            '.md': 'markdown', '.txt': 'text', '.json': 'json',
            '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
            '.xml': 'xml', '.html': 'html', '.css': 'css', '.scss': 'scss',
            '.sql': 'sql', '.env': 'env', '.ini': 'ini', '.cfg': 'config',
        }
        ext = Path(filename).suffix.lower()
        return ext_map.get(ext, 'unknown')

    def select_context_files(self, query: str, max_tokens: int = 2000) -> List[Tuple[str, str]]:
        """
        Smart file selection for a given query:
        1. Semantic search via Retriever (existing ChromaDB)
        2. BM25 keyword match on file names + first 30 lines
        3. If query mentions a filename -> always include it
        4. Always include: NEDSTER.md, main entry file, active_files
        5. Respect token budget (tiktoken count)
        Returns list of (filepath, content) tuples
        """
        selected = []
        total_tokens = 0
        enc = tiktoken.get_encoding("cl100k_base")

        # Always include NEDSTER.md if exists
        nedster_md = self.read_nedster_md()
        if nedster_md:
            tokens = len(enc.encode(nedster_md))
            if total_tokens + tokens <= max_tokens:
                selected.append(("NEDSTER.md", nedster_md))
                total_tokens += tokens

        # Check if query mentions a specific filename
        filename_pattern = re.compile(r'[\w\.\-_/]+\.(py|js|ts|go|rs|java|c|cpp|h|hpp|rb|php|sh|md|txt|json|yaml|yml|toml|xml|html|css|sql|env)')
        mentioned_files = filename_pattern.findall(query)

        for rel_path in list(self.file_index.keys()):
            # If query mentions this file
            if any(rel_path.endswith(m) for m in mentioned_files):
                content = self._read_file(rel_path)
                if content:
                    tokens = len(enc.encode(content))
                    if total_tokens + tokens <= max_tokens:
                        selected.append((rel_path, content))
                        total_tokens += tokens

        # Add main entry files if not already selected
        entry_points = ['main.py', 'index.js', 'app.py', 'main.go', 'lib.rs', 'App.java']
        for entry in entry_points:
            if entry in self.file_index and not any(s[0] == entry for s in selected):
                content = self._read_file(entry)
                if content:
                    tokens = len(enc.encode(content))
                    if total_tokens + tokens <= max_tokens:
                        selected.append((entry, content))
                        total_tokens += tokens

        # Fill remaining budget with top files by BM25 on filename + first lines
        if total_tokens < max_tokens:
            remaining_tokens = max_tokens - total_tokens
            candidates = self._bm25_file_search(query)

            for rel_path in candidates:
                if any(s[0] == rel_path for s in selected):
                    continue

                content = self._read_file(rel_path)
                if content:
                    # Truncate if needed
                    tokens = len(enc.encode(content))
                    if tokens > remaining_tokens:
                        # Truncate to fit
                        content = enc.decode(enc.encode(content)[:remaining_tokens])
                        tokens = remaining_tokens

                    if tokens > 0:
                        selected.append((rel_path, content))
                        remaining_tokens -= tokens
                        total_tokens += tokens

                if remaining_tokens <= 0:
                    break

        return selected

    def _bm25_file_search(self, query: str, top_n: int = 10) -> List[str]:
        """Simple BM25-like search on file names and first 30 lines."""
        from rank_bm25 import BM25Okapi

        if not self.file_index:
            return []

        # Build corpus: filename + first 30 lines
        corpus = []
        paths = []
        for rel_path, (size, mtime, lang) in self.file_index.items():
            content = self._read_file(rel_path, limit_lines=30)
            text = f"{rel_path} {lang} {content or ''}"
            corpus.append(text.lower().split())
            paths.append(rel_path)

        if not corpus:
            return []

        tokenized_query = query.lower().split()
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)

        # Return top N paths by score
        scored = list(zip(paths, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in scored[:top_n] if s > 0]

    def _read_file(self, rel_path: str, limit_lines: Optional[int] = None) -> Optional[str]:
        """Read file content, optionally limiting lines."""
        try:
            filepath = self.root / rel_path
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if limit_lines:
                    lines = lines[:limit_lines]
                return ''.join(lines)
        except Exception:
            return None

    def build_context_block(self, files: List[Tuple[str, str]]) -> str:
        """
        Format files as context block:
        === FILE: path/to/file.py ===
        <content (truncated if >150 lines)>
        === END FILE ===
        """
        blocks = []
        for path, content in files:
            lines = content.split('\n')
            if len(lines) > 150:
                content = '\n'.join(lines[:150]) + '\n... [truncated]'

            blocks.append(f"=== FILE: {path} ===\n{content}\n=== END FILE ===")

        return '\n\n'.join(blocks)

    def read_nedster_md(self) -> str:
        """Read NEDSTER.md project memory. Return '' if not found."""
        nedster_path = self.root / 'NEDSTER.md'
        if nedster_path.exists():
            try:
                with open(nedster_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                pass
        return ''

    def update_nedster_md(self, new_facts: str, session_id: str = '') -> None:
        """
        Append new facts to NEDSTER.md under
        ## Session {datetime}
        Extracted by LLM at session end (like milestones.md).
        """
        from datetime import datetime

        nedster_path = self.root / 'NEDSTER.md'

        # Create if not exists
        if not nedster_path.exists():
            self._create_nedster_md()

        try:
            with open(nedster_path, 'a', encoding='utf-8') as f:
                session = session_id or datetime.now().strftime('%Y-%m-%d %H:%M')
                f.write(f"\n## Session {session}\n{new_facts}\n")
        except Exception as e:
            print(f"[ContextLoader] Failed to update NEDSTER.md: {e}")

    def _create_nedster_md(self) -> None:
        """Create NEDSTER.md template."""
        from datetime import datetime

        nedster_path = self.root / 'NEDSTER.md'

        # Auto-detect info
        language = self._detect_primary_language()
        entry_point = self._detect_entry_point()
        test_runner = self._detect_test_runner()
        deps = self._detect_dependencies()

        content = f"""# NEDSTER Project Memory
## Project: {self.root.name}
## Language: {language}
## Entry Point: {entry_point}
## Test Runner: {test_runner}
## Key Dependencies: {deps}
## Architecture Notes:
(populated by agent over time)
## Decisions:
(populated by agent over time)
## Sessions:
(populated automatically)
"""
        try:
            with open(nedster_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"[ContextLoader] Failed to create NEDSTER.md: {e}")

    def _detect_primary_language(self) -> str:
        """Detect primary language by file count."""
        lang_counts: Dict[str, int] = {}
        for path, (size, mtime, lang) in self.file_index.items():
            if lang != 'unknown':
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if not lang_counts:
            return 'unknown'
        return max(lang_counts, key=lang_counts.get)

    def _detect_entry_point(self) -> str:
        """Detect main entry point file."""
        candidates = ['main.py', 'index.js', 'app.py', 'main.go', 'lib.rs', 'manage.py']
        for c in candidates:
            if c in self.file_index:
                return c
        return '(not detected)'

    def _detect_test_runner(self) -> str:
        """Detect test runner from config files."""
        if (self.root / 'pytest.ini').exists() or (self.root / 'pyproject.toml').exists():
            return 'pytest'
        if (self.root / 'package.json').exists():
            return 'npm test'
        if (self.root / 'Makefile').exists():
            return 'make test'
        return '(not detected)'

    def _detect_dependencies(self) -> str:
        """Detect key dependencies from requirements.txt or package.json."""
        deps = []

        req_path = self.root / 'requirements.txt'
        if req_path.exists():
            try:
                with open(req_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            deps.append(line.split('==')[0].split('>=')[0])
                            if len(deps) >= 5:
                                break
            except Exception:
                pass

        return ', '.join(deps[:5]) if deps else '(not detected)'
