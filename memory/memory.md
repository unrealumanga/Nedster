# Nedster Memory — Hermes Agent

## User Profile
- **User:** mnm
- **Workspace:** `/home/mnm/AI_Lab/Workspace/Nedster/`
- **Context Strategy:** Use local `.json`/`.md` files to extend context window
- **Session Style:** Resume-able via state.json injection

## Tools Available
- Python3, Node, curl, wget, git
- Tavily API: `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` (dev key)
- ClawBrowser: `/home/mnm/AI_Lab/Workspace/Nedster/sidekicks/clawbrowser/`

## Key Conventions
1. Save long-term state to `memory/state.json`
2. Write notes/summaries to `memory/*.md`
3. Load state at session start via memory injection
4. Tavily for web research (no browser needed)
5. Local files for persistent context

## Active Projects
- _(none yet — add as we go)_

## TODO
- [ ] Configure Tavily API in backend
- [ ] Set up ClawBrowser MCP tool
- [ ] Create memory injection workflow
- [ ] Test file read/write cycle

---
*Last updated: 2026-04-21*
