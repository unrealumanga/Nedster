    def _execute_response(
        self, response: str, messages: List[dict], think: bool, user_input: str
    ) -> tuple[str, list]:
        """
        Parse and execute tool calls and edit blocks.
        Max 10 iterations via IterationBudget.

        FIX: The original code created a ThreadPoolExecutor but never submitted
        futures to it — `futures` was always an empty dict, so the
        `as_completed(futures)` loop did nothing. Additionally, tools NOT in
        SAFE_PARALLEL fell into the `else` branch which only printed a warning.
        Result: no tool ever produced output to tool_msg_accumulator.

        FIXED APPROACH:
        - All tools execute via self.executor.execute() immediately.
        - SAFE_PARALLEL tools still run synchronously for simplicity
          (parallelism was broken anyway; real parallel support requires
          submitting to futures dict first).
        - Every result is accumulated into tool_msg_accumulator.
        """
        budget = IterationBudget(max_iters=10, max_chars=12000)
        final_response = response
        seen_tool_calls = set()
        applied_edits = []
        WATCHDOG.start()

        try:
            tool_loops = 0
            while budget.remaining > 0:
                iteration = budget._iters + 1
                tool_loops += 1
                self.tool_stats["loops"] += 1
                _seen_this_iteration = set()

                # Parse edit blocks
                edits = self.editor.parse_edit_blocks(response)

                # Parse tool calls
                tool_calls = parse_tool_calls(response)

                if not edits and not tool_calls:
                    break

                # Execute edits
                for edit in edits:
                    result = self.editor.apply_edit(edit, auto=self.auto)
                    self.tui.print_status(f"Edit: {result}")
                    if "Edited" in result or "Created" in result or "Overwritten" in result:
                        self.tool_stats["edits"] += 1
                        applied_edits.append(edit)

                    # Auto-check syntax for Python files
                    if edit.get("path", "").endswith(".py"):
                        syntax_result = self._check_file_syntax(str(edit.get("path", "")))
                        if syntax_result != "OK":
                            self.tui.print_warning(f"Syntax issue: {syntax_result}")

                tool_msg_accumulator = ""

                # Execute tool calls
                import json
                from tools import TOOL_REGISTRY, TOOL_NAME_ALIASES, normalize_tool_args

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    args = normalize_tool_args(tool_name, tool_call.get("args", {}))

                    raw_name = tool_name
                    # Normalize tool name
                    t_name = raw_name.strip().lower().replace("-", "_")
                    if t_name not in TOOL_REGISTRY:
                        aliased = TOOL_NAME_ALIASES.get(t_name)
                        if aliased is None and t_name in TOOL_NAME_ALIASES:
                            continue  # explicitly discarded
                        if aliased and aliased in TOOL_REGISTRY:
                            t_name = aliased
                        else:
                            self.tui.print_status(
                                f"[BLOCKED] '{raw_name}' not a valid tool", "bold red"
                            )
                            tool_msg_accumulator += (
                                f"[ERROR: '{raw_name}' unknown. Use write_file to create files.]\n"
                            )
                            continue

                    tool_name = t_name  # use normalized name

                    call_hash = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

                    NEVER_DEDUP = {"list_dir", "git_status", "run_bash", "read_file"}
                    if tool_name not in NEVER_DEDUP:
                        if call_hash in _seen_this_iteration:
                            self.tui.print_status(
                                f"[SKIP] Identical call in same batch: {tool_name}", "dim"
                            )
                            continue
                        _seen_this_iteration.add(call_hash)

                    self.tui.print_tool_call(name=tool_name, args=args)

                    # FIX: Execute ALL tools through executor and collect results.
                    # Previously SAFE_PARALLEL tools ran but result was discarded;
                    # non-SAFE_PARALLEL tools only printed a warning and never ran.
                    if tool_name in TOOL_REGISTRY:
                        # Inject cwd for git tools
                        if "cwd" not in args and tool_name.startswith("git_"):
                            args["cwd"] = self.project_dir

                        result = self.executor.execute(tool_name, args, self.tui)
                        self.tui.print_tool_result(tool_name, result, verbose=self.verbose)
                        self.tool_stats["calls"] += 1
                        tool_msg_accumulator += f"[Tool result: {tool_name}]\n{result}\n\n"
                        WATCHDOG.ping()
                    else:
                        # Should not happen after alias resolution above, but guard anyway
                        self.tui.print_warning(f"Unknown tool after normalization: {tool_name}")
                        tool_msg_accumulator += f"[ERROR: '{tool_name}' not in registry]\n"

                # If nothing ran at all, break
                if not tool_msg_accumulator and not edits:
                    break

                # Regenerate response with tool results
                if tool_msg_accumulator:
                    preview = (
                        tool_msg_accumulator[:70]
                        .replace("\n", " ")
                        .replace("[Tool result: ", "")
                        .replace("]", "")
                    )
                    self.tui.print_status(
                        f"  • [{self.tui.COLORS['tool']}]Result: {preview}...[/]", ""
                    )

                    if not budget.consume(messages):
                        tool_msg_accumulator += budget.inject_limit_message()
                        break

                    tool_msg_accumulator += _build_verification_injection(tool_msg_accumulator)
                    tool_msg_accumulator += "\nContinue."

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": tool_msg_accumulator})
                    response = self._stream_generate(messages, think=think)
                    final_response += "\n\n" + response
                else:
                    break

                # After edits, offer to run tests
                if edits and iteration == 1:
                    test_runner = self._detect_test_runner()
                    if test_runner != "unknown":
                        self.tui.print_status(f"Test runner detected: {test_runner}")
                        if not self.auto:
                            try:
                                resp = input("Run tests? [y/N] ").strip().lower()
                                if resp in ("y", "yes"):
                                    test_result = self._run_tests()
                                    self.tui.print_status(
                                        f"Tests: {test_result[:200]}..."
                                    )
                                    self.tool_stats["tests"] += 1
                            except (EOFError, KeyboardInterrupt):
                                pass

        finally:
            WATCHDOG.stop()

        verification_warning = _verify_task_completion(user_input, final_response)
        if verification_warning:
            print(verification_warning)
            messages.append({"role": "assistant", "content": final_response})
            messages.append(
                {
                    "role": "user",
                    "content": verification_warning
                    + "\nDo NOT confirm success. Fix the incomplete task now.",
                }
            )
            final_response += "\n\n" + self._stream_generate(messages, think=think)

        return final_response, applied_edits
