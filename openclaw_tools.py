import glob
import os
import re

def glob_search(pattern: str, path: str = ".") -> str:
    """Find files by name patterns like **/*.js"""
    from tools import SESSION, _resolve_path
    actual_path = _resolve_path(str(path))
    if not os.path.isdir(actual_path):
        return f"Error: Directory {actual_path} not found"
        
    search_pattern = os.path.join(actual_path, pattern)
    try:
        files = glob.glob(search_pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        # Make paths relative to active project dir
        rel_files = [os.path.relpath(f, SESSION.active_project_dir) for f in files]
        
        if not rel_files:
            return f"No files matched pattern '{pattern}'"
        
        return "Matches:\\n" + "\\n".join(sorted(rel_files)[:50])
    except Exception as e:
        return f"Error globbing: {e}"

def grep_search(pattern: str, include: str = "*", path: str = ".") -> str:
    """Fast content search using regex via ripgrep (if available) or basic grep"""
    from tools import run_bash, _resolve_path
    import subprocess
    
    actual_path = _resolve_path(str(path))
    # Check if rg exists
    has_rg = subprocess.run(["which", "rg"], capture_output=True).returncode == 0
    
    try:
        if has_rg:
            inc_flag = f"-g '{include}'" if include and include != "*" else ""
            cmd = f"rg -n {inc_flag} '{pattern}' '{actual_path}' | head -n 50"
        else:
            inc_flag = f"--include='{include}'" if include and include != "*" else ""
            cmd = f"grep -rnE {inc_flag} '{pattern}' '{actual_path}' | head -n 50"
        
        return run_bash(cmd)
    except Exception as e:
        return f"Error searching: {e}"

def edit_file(path: str, oldString: str, newString: str, replaceAll: bool = False) -> str:
    """Exact string replacement in a file."""
    from tools import _resolve_path, SESSION
    
    actual_path = _resolve_path(str(path))
    if not os.path.exists(actual_path):
        return f"Error: File {actual_path} does not exist. Use write_file to create new files."
        
    try:
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if oldString not in content:
            return "Error: oldString not found in content. Check whitespace and indentation carefully."
            
        matches = content.count(oldString)
        if matches > 1 and not replaceAll:
            return "Error: Found multiple matches for oldString. Use replaceAll=true or provide more surrounding lines."
            
        new_content = content.replace(oldString, newString) if replaceAll else content.replace(oldString, newString, 1)
        
        with open(actual_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Successfully edited {actual_path} (replaced {matches if replaceAll else 1} occurrences)."
    except Exception as e:
        return f"Error editing file: {e}"

def web_fetch(url: str, format: str = "markdown") -> str:
    """Fetch content from a URL and optionally convert to markdown."""
    import subprocess
    try:
        if format == "markdown":
            # Attempt to use trafilatura or beautifulsoup if available, fallback to curl + pandoc or html2text
            # Just use curl and strip basic HTML for now
            res = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
            html = res.stdout
            
            # Simple strip
            from html.parser import HTMLParser
            class HTMLFilter(HTMLParser):
                text = ""
                def handle_data(self, data):
                    self.text += data
            f = HTMLFilter()
            f.feed(html)
            text = f.text.strip()
            
            if len(text) > 4000:
                text = text[:4000] + "\\n...[truncated]"
            return text
        else:
            res = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
            text = res.stdout
            if len(text) > 4000:
                text = text[:4000] + "\\n...[truncated]"
            return text
    except Exception as e:
        return f"Error fetching URL: {e}"
        
def todowrite(todos: list) -> str:
    """Create and manage a structured task list."""
    from tools import SESSION
    import json
    
    todo_path = os.path.join(SESSION.active_project_dir, ".nedster_todos.json")
    try:
        with open(todo_path, "w", encoding="utf-8") as f:
            json.dump(todos, f, indent=2)
        return f"Successfully updated todo list at {todo_path}"
    except Exception as e:
        return f"Error updating todo list: {e}"
