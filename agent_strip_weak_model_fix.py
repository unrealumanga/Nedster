    def _strip_weak_model_artifacts(self, text: str) -> str:
        """
        Strip artifacts leaked by weak/chat-only models that hallucinate
        system-prompt fragments into their output.

        FIX: The original had orphaned code (lines checking `accumulated` and
        `_REPEAT_THRESHOLD`) placed AFTER `return text.strip()`. That dead code
        referenced an undefined variable and would have raised NameError if
        somehow reached. Removed entirely.
        """
        import re
        text = re.sub(r'<tool\s+name="[^"]*">.*?</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[YOU ARE NEDSTER\..*?\]', '', text, flags=re.DOTALL)
        text = re.sub(r'YOU ARE NEDSTER[.,].*?(?=\n|$)', '', text, flags=re.MULTILINE)
        text = re.sub(r'={3,}\s*FILE:.*?={3,}', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*Final response:\*\*\s*', '', text)
        text = re.sub(r'\*\*Final reply:\*\*\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        # Strip leaked system-prompt fragments from chat-only models
        text = re.sub(r'DIRECTIVE (?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT).*?(?=\n\n|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'\[NON-NEGOTIABLE EXECUTION RULES\].*?(?=\n\n|$)', '', text, flags=re.DOTALL)
        return text.strip()
