"""
Mock configurabile per test senza server LLM.
"""
import time

from Carlo.model.llm_client import LLMClient


class LLMClientMock(LLMClient):
    """
    Client mock per testing. Supporta:
    - Modalità echo (default): ritorna il prompt con prefisso
    - Modalità dizionario: risposte predefinite per pattern specifici
    - Delay opzionale per simulare latenza di rete
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        delay: float = 0.0,
        default_response: str | None = None,
    ):
        """
        Args:
            responses: Dizionario pattern → risposta. Se il pattern è contenuto
                       nel prompt, viene ritornata la risposta corrispondente.
            delay: Secondi di attesa prima di ritornare (simula latenza).
            default_response: Risposta di default se nessun pattern matcha.
                              Se None, usa modalità echo.
        """
        self._responses = responses or {}
        self._delay = delay
        self._default_response = default_response

    def generate(self, prompt: str) -> str:
        """
        Genera una risposta mock.
        
        Args:
            prompt: Il prompt ricevuto.
        
        Returns:
            Risposta mock basata su pattern o echo.
        """
        if self._delay > 0:
            time.sleep(self._delay)

        # Cerca match nei pattern configurati
        for pattern, response in self._responses.items():
            if pattern.lower() in prompt.lower():
                return response

        # Fallback: risposta default o echo
        if self._default_response is not None:
            return self._default_response

        # Echo mode: ritorna versione troncata del prompt
        truncated = prompt[:100] + "..." if len(prompt) > 100 else prompt
        return f"[MOCK] {truncated}"
