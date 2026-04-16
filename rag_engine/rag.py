# File: rag.py
import sys
import os
import ollama
from datetime import datetime
from rag_engine.retriever import Retriever
from memory import MemoryManager
from personality import load_personality, build_system_prompt
from journal import (
    detect_topic, score_session,
    write_journal_entry, update_index, update_project_doc,
    log_decision, capture_research, save_code_snippet,
    trigger_ingest_if_needed, search_journal
)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.model = os.environ.get("MODEL", "qwen3.5:9b")
        self.memory = MemoryManager(llm_model_name=self.model)
        self.personality = load_personality()
        self.tool_stats = {"calls": 0, "loops": 0}
        print(self.personality.get("greeting", "Ready."))

        self.session_file = "session_memory.md"

        # Initialize or clear session file on start
        with open(self.session_file, "a") as f:
            f.write(
                f"\n\n# New Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        self._load_milestones_into_context()
        self.session_topic = "general"
        self.session_project = None
        self.session_research = []
        self.session_decisions = []
        self.session_code = []
        self.session_findings = []
        self.tasks_completed = 0

        self._load_active_projects()

    def _load_milestones_into_context(self):
        path = os.path.expanduser("~/.aria/milestones.md")
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            recent = "".join(lines[-60:]).strip()
            if recent:
                self.memory.session_summary = (
                    "[Long-term memory from previous sessions]\n" + recent
                )
                print(f"[Milestones loaded: {len(lines)} lines]")

    def _save_session_milestones(self):
        if not self.memory.short_term:
            return
        history = self.memory.get_last_n_turns_text(n=10)
        prompt = (
            "Extract key facts, fixes, decisions, and user preferences from this "
            "conversation as bullet points.\n"
            "Format each: [FACT/FIX/SETUP/PREF] description\n"
            "Max 8 bullets. Specific. No preamble.\n\n"
            f"Conversation:\n{history}\nMilestones:"
        )
        try:
            resp = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_ctx": 2048,
                    "num_predict": 200,
                    "temperature": 0.0,
                    "think": False,
                },
            )
            milestones = resp["response"].strip()
            os.makedirs(os.path.expanduser("~/.aria"), exist_ok=True)
            with open(os.path.expanduser("~/.aria/milestones.md"), "a") as f:
                f.write(
                    f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                    f"Session {self.memory.session_id}\n{milestones}\n"
                )
            print("[Milestones saved]")
            
            quality = score_session(
                self.memory.short_term,
                tasks_completed=self.tasks_completed
            )
            narrative_prompt = (
                "Write a 3-sentence summary of this session for a personal journal.\n"
                "Focus on: what was accomplished, what was decided, what was learned.\n"
                f"Session topic: {self.session_topic}\n"
                f"Conversation:\n{self.memory.get_last_n_turns_text(n=8)}\nSummary:"
            )
            try:
                resp = ollama.generate(model=self.model, prompt=narrative_prompt,
                    options={"num_ctx": 2048, "num_predict": 150,
                             "temperature": 0.0, "think": False})
                narrative = resp["response"].strip()
            except Exception:
                narrative = f"Session on {self.session_topic}."

            findings = [milestones.strip()] if milestones else []

            import re as _re
            open_qs = []
            for msg in self.memory.short_term[-6:]:
                if msg['role'] == 'assistant':
                    qs = _re.findall(r'[A-Z][^.!?]*\?', msg['content'])
                    open_qs.extend(qs[:2])
            open_qs = list(set(open_qs))[:5]

            entry_path = write_journal_entry(
                session_id=self.memory.session_id,
                topic=self.session_topic,
                project=self.session_project,
                summary=narrative[:100],
                decisions=self.session_decisions,
                research_items=self.session_research,
                code_files=self.session_code,
                narrative=narrative,
                findings=findings,
                open_questions=open_qs,
                quality=quality,
            )

            if entry_path:
                update_index(
                    session_id=self.memory.session_id,
                    topic=self.session_topic,
                    project=self.session_project,
                    summary=narrative[:60],
                    entry_path=entry_path,
                )
                print(f"[Journal saved: {entry_path.name}]")
                import os
                trigger_ingest_if_needed(os.getcwd())
        except Exception as e:
            print(f"[Milestone save failed: {e}]")

    def log_to_file(self, query, response):
        with open(self.session_file, "a") as f:
            f.write(f"\n**User:** {query}\n\n**Assistant:** {response}\n\n---\n")

    def rewrite_query(self, query):
        history_text = self.memory.get_last_n_turns_text(n=4)
        if not history_text:
            return query

        prompt = (
            "You are a search query rewriting AI.\n"
            "Given the conversation history and a new user question, rewrite the question "
            "into a single, standalone search query that includes all necessary context from the history.\n"
            "DO NOT answer the question. ONLY output the rewritten search query.\n\n"
            "History:\n"
        )

        prompt += history_text + f"\nNew Question: {query}\nRewritten Query:"

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_ctx": 1024,
                    "num_predict": 40,
                    "temperature": 0.0,
                    "think": False,
                },
            )
            rewritten = response["response"].strip()
            # Remove quotes if the LLM wrapped it
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            print(f"[\033[90mQuery Rewritten for Search: {rewritten}\033[0m]")
            return rewritten if rewritten else query
        except Exception:
            return query

    def plan_and_execute(self, query, think=False):
        prompt = (
            "Break this task into 3-5 numbered steps. Output ONLY the steps, no explanation:\n"
            f"{query}"
        )
        try:
            print("[Planner Agent] Generating plan...")
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_ctx": 1024,
                    "num_predict": 120,
                    "temperature": 0.0,
                },
            )
            plan_text = response["response"].strip()

            steps = [s.strip() for s in plan_text.split("\n") if s.strip()]
            n = len(steps)
            print(f"[Planner Agent] Plan created with {n} steps.")

            for i, step in enumerate(steps):
                print(f"\n[STEP {i + 1}/{n}]: {step}")
                self.generate(step, think=think)

        except Exception as e:
            print(f"[Planner error: {e}]")


    def _detect_and_store_keys(self, query: str) -> str:
        """
        If user pastes an API key in chat, store it securely and
        confirm WITHOUT echoing the key. Return masked confirmation or "".
        """
        import re
        # Tavily key pattern
        tavily_pattern = re.compile(r'\btvly-[a-zA-Z0-9\-_]{20,}\b')
        match = tavily_pattern.search(query)
        if match:
            key = match.group(0)
            try:
                from tools import store_tavily_key, probe_tools
                store_tavily_key(key)
                # Remove key from query before sending to LLM
                sanitized_query = query.replace(key, "[TAVILY_KEY_STORED]")
                return sanitized_query
            except Exception:
                return query.replace(key, "[KEY_DETECTED_STORE_FAILED]")
        return query

    def _resolve_references(self, query: str) -> str:
        """
        Resolve pronouns and references using recent session context.
        "which one?" → looks at last comparison in session memory
        "it" → looks at last named entity in session
        "that" → looks at last specific thing mentioned
        """
        q_lower = query.lower().strip().rstrip("?!")

        REFERENCE_TRIGGERS = {
            "which one", "which is better", "which is faster",
            "which wins", "compare", "vs", "faster", "slower",
            "better", "worse", "the best", "the winner"
        }

        if not any(t in q_lower for t in REFERENCE_TRIGGERS):
            return query

        # Search recent memory for last comparison
        history_text = self.memory.get_last_n_turns_text(n=6)
        if not history_text:
            return query

        # Find pairs of things being compared in history
        import re
        # Look for "X vs Y", "X or Y", "comparing X and Y"
        pairs = re.findall(
            r'(\b\w+(?:\s+\w+){0,2})\s+(?:vs|versus|or|compared to)\s+(\b\w+(?:\s+\w+){0,2})',
            history_text, re.IGNORECASE
        )

        if pairs:
            a, b = pairs[-1]  # most recent comparison
            resolved = f"{query} [comparing: {a.strip()} vs {b.strip()}]"
            return resolved

        return query

    def _build_context(self, top_docs: list, web_results: str = "") -> tuple:
        """
        Returns (rag_context_str, web_context_str) separately.
        RAG gets 1800 tokens budget. Web gets 800 tokens budget.
        Never mix them — keeps RAG clean from web contamination.
        """
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        rag_parts = []
        rag_budget = 8000
        used = 0
        for doc_text, score, meta in top_docs:
            if score < 0.20:  # skip low-relevance chunks (relaxed for code files)
                continue
            chunk = (f"[Source: {meta.get('source_file','?')}, "
                     f"Score: {score:.2f}]\n{doc_text}\n---")
            chunk_tokens = len(enc.encode(chunk))
            if used + chunk_tokens > rag_budget:
                break
            rag_parts.append(chunk)
            used += chunk_tokens

        rag_str = "\n".join(rag_parts) if rag_parts else ""

        # Web results get their own capped budget
        web_str = ""
        if web_results:
            web_tokens = enc.encode(web_results)
            if len(web_tokens) > 800:
                web_results = enc.decode(web_tokens[:800]) + "...[truncated]"
            web_str = f"[Web Search Results]\n{web_results}"

        return rag_str, web_str

    def _classify_input(self, query: str) -> str:
        """Classify H2's input to control response style."""
        q = query.strip()
        q_lower = q.lower().rstrip("!?.")

        POKES = {"so", "?", "!?", "so?", "so!>?", "well?", "then?",
                 "and?", "continue", "go on", "do it", "what happened",
                 "and then", "go", "next", "proceed"}
        IDENTITIES = {"h2 here", "h2", "hello", "hi", "hey", "morning",
                      "good morning", "wake up"}
        DECISIONS = {"yes", "yes!", "all of them", "nope", "skip it",
                     "no", "ok", "sure", "sounds good", "perfect"}
        EMOTIONAL = {":)", ":(", ":D", "lol", "haha", "damn", "nice",
                     "great", "wow", "ugh", "ffs"}

        word_count = len(q.split())

        if q_lower in POKES or (word_count <= 3 and q.endswith("?")):
            return "poke"
        if q_lower in IDENTITIES:
            return "identity"
        if q_lower in DECISIONS:
            return "decision"
        if q_lower in EMOTIONAL or (word_count == 1 and not q[0].isalpha()):
            return "emotional"
        if word_count <= 5:
            return "short"
        if word_count <= 20:
            return "medium"
        return "long"

    def generate(self, query, think=False):
        query = self._detect_and_store_keys(query)
        classification = self._classify_input(query)
        query = self._resolve_references(query)
        if self.memory.turn_count == 1 and self.session_topic == "general":
            msgs = self.memory.short_term[-2:] if self.memory.short_term else []
            all_msgs = msgs + [{"role": "user", "content": query}]
            self.session_topic = detect_topic(all_msgs)
            print(f"[\033[36mSession topic: {self.session_topic}\033[0m]")

        if classification == "poke":
            # Skip RAG, skip rewrite, use only session context
            top_docs = []
            rag_str = ""
            web_str = ""
            # Inject pending plan from last assistant message if exists
            pending = ""
            if self.memory.short_term:
                last_assistant = next(
                    (m["content"] for m in reversed(self.memory.short_term)
                     if m["role"] == "assistant"), ""
                )
                if last_assistant:
                    pending = f"[Your last stated plan: {last_assistant[:300]}]\n"
            search_query = query
            # We construct user_message later, but we need search_query defined
        elif classification in ("identity", "emotional", "decision"):
            top_docs = []
            rag_str = ""
            web_str = ""
            search_query = query
            # No rewrite, no retrieval
        elif classification == "short" and len(query.strip()) < 8:
            top_docs = []
            rag_str = ""
            web_str = ""
            search_query = query
        else:
            search_query = self.rewrite_query(query)
            top_docs = self.retriever.retrieve(search_query)

            # Inject chunks
            context_parts = []
            for doc_text, score, meta in top_docs:
                filename = (
                    meta.get("source_file", "Unknown")
                    if meta and isinstance(meta, dict)
                    else "Unknown"
                )
                chunk_injection = (
                    f"[Source: {filename}, Score: {score:.2f}]\n{doc_text}\n---"
                )
                context_parts.append(chunk_injection)
            context_str = "\n".join(context_parts)

        # System prompt
        base_instructions = (
            "Answer using ONLY the provided context and conversation history. "
            "You MUST answer entirely in English. "
            "If the answer is not in the context or history, say 'I don't have that information.' "
            "Cite the source filename for each fact you use."
        )
        system_prompt = build_system_prompt(self.personality, base_instructions)

        # Add live tool status to system prompt
        tool_lines = []
        for tool, status in getattr(self, "_tool_inventory", {}).items():
            if status == "OK":
                tool_lines.append(f"  - {tool}: ACTIVE")
            elif "NO_KEY" in status:
                tool_lines.append(f"  - {tool}: needs key")
        if tool_lines:
            tool_section = "ACTIVE TOOLS THIS SESSION:\n" + "\n".join(tool_lines)
            system_prompt = system_prompt + "\n\n" + tool_section

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            mem_msgs = self.memory.get_context_messages()
            mem_tokens = sum(len(enc.encode(m["content"])) for m in mem_msgs)
            ctx_tokens = len(enc.encode(context_str)) if context_str else 0
            sys_tokens = len(enc.encode(system_prompt))
            total = sys_tokens + mem_tokens + ctx_tokens
            budget = int(os.environ.get("TURBOQUANT_CONTEXT_SIZE", "262144"))
            if total > budget * 0.80:
                print(f"[Context: {total}/{budget} tokens — trimming RAG]")
                top_docs = top_docs[:2]
                context_str = "\n".join(
                    f"[Source: {m.get('source_file', '?') if isinstance(m, dict) else '?'}, Score: {s:.2f}]\n{d}\n---"
                    for d, s, m in top_docs
                )
            if total > budget * 0.95:
                context_str = "[Context trimmed: token budget exceeded]"
                print("[Emergency context trim]")
        except Exception:
            pass  # tiktoken unavailable — skip tracking

        # Build messages: system -> past session turns -> current query with context
        messages = [{"role": "system", "content": system_prompt}]

        for past_msg in self.memory.get_context_messages():
            messages.append(past_msg)

        if classification == "poke":
            user_message = f"[EXECUTE NOW — poke received]\n{pending}Current input: {query}"
        else:
            LENGTH_DIRECTIVES = {
                "poke":     "[EXECUTE PLAN NOW. Report result in 2 lines max.]",
                "identity": "[1-line warm acknowledgment only. Wait for next input.]",
                "decision": "[Execute the chosen option. 1-line confirmation after.]",
                "emotional": "[1-line acknowledgment. Then 1 relevant action or question.]",
                "short":    "[Short reply. Max 3 lines unless technical output needed.]",
                "medium":   "[Concise reply. Use prose not bullets for non-technical answers.]",
                "long":     "[Full reply allowed. Use bullets only if listing 4+ distinct items.]",
            }
            directive = LENGTH_DIRECTIVES.get(classification, "")
            sections = []
            if rag_str:
                sections.append(f"Retrieved Context:\n{rag_str}")
            if web_str:
                sections.append(f"Live Web Data:\n{web_str}")
            sections.append(f"Current Question: {query}")
            user_message = f"{directive}\n\n" + "\n\n".join(sections)

        messages.append({"role": "user", "content": user_message})

        # Ollama tip: keep Ollama server warm (don't stop between queries)
        # Context tip: --think flag adds ~500 reasoning tokens, disable for fast answers
        # Temperature 0.1 strongly anchors the model to English and prevents hallucination
        options = {"num_ctx": int(os.environ.get("TURBOQUANT_CONTEXT_SIZE", "262144")), "temperature": 0.1}
        if not think:
            options["think"] = False  # skip chain-of-thought to save tokens

        try:
            print(f"Generating answer (Think Mode: {think})...\n")
            
            # Check if TurboQuant server should be used via OpenAI API
            use_turboquant = os.environ.get("USE_TURBOQUANT", "0") == "1"
            
            if use_turboquant:
                try:
                    import openai
                    print("[\033[36mRunning via TurboQuant KV Compression Server\033[0m]")
                    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="sk-turboquant")
                    # convert messages for OpenAI
                    oai_msgs = []
                    for m in messages:
                        oai_msgs.append({"role": m["role"], "content": m["content"]})
                    
                    response_stream = client.chat.completions.create(
                        model=self.model,
                        messages=oai_msgs,
                        stream=True,
                        temperature=0.1
                    )
                    
                    # mock Ollama stream response format
                    def turboquant_stream_adapter(stream):
                        for chunk in stream:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield {"message": {"content": chunk.choices[0].delta.content}}
                                
                    response = turboquant_stream_adapter(response_stream)
                except ImportError:
                    print("Please run: pip install openai")
                    use_turboquant = False
            
            if not use_turboquant:
                response = ollama.chat(
                    model=self.model, messages=messages, stream=True, options=options
                )

            import re
            # State machine to strip <think>...</think> and emoji from stream
            _think_depth = 0
            _think_buf = ""
            _EMOJI_RE = re.compile(
                "[\U0001F300-\U0001FFFF\U00002600-\U000027BF\U0001F900-\U0001F9FF]"
            )
            _token_count = 0

            for chunk in response:
                raw = chunk["message"]["content"]

                # Buffer-based think tag stripper
                _think_buf += raw
                visible = ""

                while _think_buf:
                    if _think_depth == 0:
                        open_pos = _think_buf.find("<think>")
                        if open_pos == -1:
                            # No open tag — safe to print everything
                            visible += _think_buf
                            _think_buf = ""
                        else:
                            # Print before the tag, then enter think mode
                            visible += _think_buf[:open_pos]
                            _think_buf = _think_buf[open_pos + 7:]
                            _think_depth = 1
                    else:
                        close_pos = _think_buf.find("</think>")
                        if close_pos == -1:
                            # Still inside think block — buffer it, don't print
                            break
                        else:
                            # Skip think block content
                            _think_buf = _think_buf[close_pos + 8:]
                            _think_depth = 0

                # Strip emoji from visible output (backstop for Modelfile rule)
                visible = _EMOJI_RE.sub("", visible)

                if visible:
                    print(visible, end="", flush=True)
                    full_response += visible
                    _token_count += len(visible.split())

            # After stream: show token count
            print(f"\n[\033[90m~{_token_count} tokens | ctx {_token_count*100//int(os.environ.get('TURBOQUANT_CONTEXT_SIZE', '262144'))}%\033[0m]")

            # Strip any remaining think tags from full_response for memory storage
            full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

            import tools

            MAX_TOOL_LOOPS = 3
            tool_loops = 0

            complete_response = full_response

            DECISION_SIGNALS = [
                'decided', "don't", "won't", 'never', 'always', 'from now on',
                'quarantine', 'stop', 'archive', 'delete', 'switch to',
                'use only', 'no more', 'permanently'
            ]
            query_lower = query.lower()
            if any(sig in query_lower for sig in DECISION_SIGNALS):
                decision_text = query[:200].strip()
                log_decision(
                    decision=decision_text,
                    context=full_response[:300],
                    project=self.session_project
                )
                self.session_decisions.append(decision_text)
                print(f"[[35mDecision logged: {decision_text[:60]}...[0m]")

            from tools import WATCHDOG
            WATCHDOG.start()
            while tool_loops < MAX_TOOL_LOOPS:
                tool_calls = tools.parse_tool_calls(full_response)
                if not tool_calls:
                    break

                tool_loops += 1
                self.tool_stats["loops"] += 1
                self.tool_stats["calls"] += len(tool_calls)

                tool_results_str = ""
                for tc in tool_calls:
                    t_name = tc["name"]
                    t_args = tc.get("args", {})
                    if t_name in tools.TOOL_REGISTRY:
                        try:
                            res = tools.TOOL_REGISTRY[t_name](**t_args)
                            tool_results_str += f"Tool {t_name} output:\n{res}\n"
                        except Exception as e:
                            tool_results_str += f"Tool {t_name} error:\n{e}\n"
                    else:
                        tool_results_str += f"Tool {t_name} not found.\n"
                WATCHDOG.ping()
                messages.append({"role": "assistant", "content": full_response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool Results:\n{tool_results_str}\n\nContinue.",
                    }
                )

                if use_turboquant:
                    oai_msgs.append({"role": "assistant", "content": full_response})
                    oai_msgs.append({"role": "user", "content": f"Tool Results:\n{tool_results_str}\n\nContinue."})
                    response_stream = client.chat.completions.create(
                        model=self.model,
                        messages=oai_msgs,
                        stream=True,
                        temperature=0.1
                    )
                    response = turboquant_stream_adapter(response_stream)
                else:
                    response = ollama.chat(
                        model=self.model, messages=messages, stream=True, options=options
                    )

                import re
                # State machine to strip <think>...</think> and emoji from stream
                _think_depth = 0
                _think_buf = ""
                _EMOJI_RE = re.compile(
                    "[\U0001F300-\U0001FFFF\U00002600-\U000027BF\U0001F900-\U0001F9FF]"
                )
                _token_count = 0
    
                for chunk in response:
                    raw = chunk["message"]["content"]
    
                    # Buffer-based think tag stripper
                    _think_buf += raw
                    visible = ""
    
                    while _think_buf:
                        if _think_depth == 0:
                            open_pos = _think_buf.find("<think>")
                            if open_pos == -1:
                                # No open tag — safe to print everything
                                visible += _think_buf
                                _think_buf = ""
                            else:
                                # Print before the tag, then enter think mode
                                visible += _think_buf[:open_pos]
                                _think_buf = _think_buf[open_pos + 7:]
                                _think_depth = 1
                        else:
                            close_pos = _think_buf.find("</think>")
                            if close_pos == -1:
                                # Still inside think block — buffer it, don't print
                                break
                            else:
                                # Skip think block content
                                _think_buf = _think_buf[close_pos + 8:]
                                _think_depth = 0
    
                    # Strip emoji from visible output (backstop for Modelfile rule)
                    visible = _EMOJI_RE.sub("", visible)
    
                    if visible:
                        print(visible, end="", flush=True)
                        full_response += visible
                        _token_count += len(visible.split())
    
                # After stream: show token count
                print(f"\n[\033[90m~{_token_count} tokens | ctx {_token_count*100//int(os.environ.get('TURBOQUANT_CONTEXT_SIZE', '262144'))}%\033[0m]")
    
                # Strip any remaining think tags from full_response for memory storage
                full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                complete_response += "\n\n" + full_response

            WATCHDOG.stop()
            self.memory.add_turn(query, complete_response)

            # Log to markdown file
            self.log_to_file(query, complete_response)

            # Save to ChromaDB Long-Term Memory
            self.retriever.add_to_memory(
                query, complete_response, session_id=self.memory.session_id
            )

        except ollama.ResponseError as e:
            if "not found" in str(e).lower():
                print(f"\nModel {self.model} not found. Re-building automatically...")
                import subprocess

                subprocess.run(["ollama", "pull", "qwen3.5:9b"])
                subprocess.run(["ollama", "create", "aria-qwen", "-f", "./Modelfile"])
                print("\nModel built. Please try your query again.")
        except Exception as e:
            err_str = str(e).lower()
            if "connection refused" in err_str:
                print("\nError: Ollama not running. Run: ollama serve")
            else:
                print(f"\nGeneration error: {e}")
