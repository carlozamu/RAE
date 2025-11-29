class LLMClient:
    """
    Minimal LLM client interface. Replace `generate` with a real call when available.
    """

    def generate(self, prompt: str) -> str:
        # Stubbed behavior: return a mock answer so the "Answer" section is not empty.
        return f"[LLM stub answer] {prompt}\n[stub output] Questo è un output fittizio per i test end-to-end."
