import re

with open("tools.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add the new tools code right before TOOL_REGISTRY updates
openclaw_code = '''
def glob_search(pattern: str, path: str = ".") -> str:
    """Find files by name patterns like **/*.js"""
    import glob, os
    actual_path = _resolve_path(str(path))
    if not os.path.isdir(actual_path):
        return f"Error: Directory {actual_path} not found"
        
    search_pattern = os.path.join(actual_path, pattern)
    try:
        files = glob.glob(search_pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        rel_files = [os.path.relpath(f, SESSION.active_project_dir) for f in files]
        
        if not rel_files:
            return f"No files matched pattern '{pattern}'"
        
        return "Matches:\\n" + "\\n".join(sorted(rel_files)[:50])
    except Exception as e:
        return f"Error globbing: {e}"

def grep_search(pattern: str, include: str = "*", path: str = ".") -> str:
    """Fast content search using regex via ripgrep (if available) or basic grep"""
    import subprocess
    actual_path = _resolve_path(str(path))
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
    import os
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
            res = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
            html = res.stdout
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
    import json, os
    todo_path = os.path.join(SESSION.active_project_dir, ".nedster_todos.json")
    try:
        with open(todo_path, "w", encoding="utf-8") as f:
            json.dump(todos, f, indent=2)
        return f"Successfully updated todo list at {todo_path}"
    except Exception as e:
        return f"Error updating todo list: {e}"
'''

# Find the end of TOOL_NAME_ALIASES dict to insert new aliases
aliases_block = '''
    # Search variants
    "grep":            "search_code",
    "find":            "search_code",
    "search":          "search_code",
'''

new_aliases_block = '''
    # Search variants
    "grep":            "grep_search",
    "glob":            "glob_search",
    "find":            "glob_search",
    "search":          "grep_search",
    
    # Edit variants
    "edit":            "edit_file",
    "edit_file":       "edit_file",
    "modify_file":     "edit_file",
    "update_file":     "edit_file",
    
    # Web variants
    "webfetch":        "web_fetch",
    "curl":            "web_fetch",
    "fetch":           "web_fetch",
    
    # Task variants
    "todowrite":       "todowrite",
    "todo":            "todowrite",
'''
text = text.replace(aliases_block, new_aliases_block)

# Add TOOL_REGISTRY updates
registry_updates = '''
TOOL_REGISTRY["write_file"] = write_file
TOOL_REGISTRY["_create_file"] = _create_file
'''

new_registry_updates = '''
TOOL_REGISTRY["glob_search"] = glob_search
TOOL_REGISTRY["grep_search"] = grep_search
TOOL_REGISTRY["edit_file"] = edit_file
TOOL_REGISTRY["web_fetch"] = web_fetch
TOOL_REGISTRY["todowrite"] = todowrite

TOOL_REGISTRY["write_file"] = write_file
TOOL_REGISTRY["_create_file"] = _create_file
'''
text = text.replace(registry_updates, new_registry_updates)

# Insert the function definitions before TOOL_REGISTRY updates
text = text.replace('TOOL_REGISTRY["glob_search"] = glob_search', openclaw_code + '\nTOOL_REGISTRY["glob_search"] = glob_search')

with open("tools.py", "w", encoding="utf-8") as f:
    f.write(text)
