"""
Adapter per utilizzare il server LLM remoto di Filippo.
Wrappa le chiamate async in sync per compatibilità con l'interfaccia LLMClient.
"""
import asyncio

from Carlo.model.llm_client import LLMClient
from Filippo.LLM import LLM


class LLMClientRemote(LLMClient):
    """
    Client che usa il server LLM esterno (vLLM/API) tramite Filippo.LLM.
    L'istanza LLM viene passata dall'esterno (singleton), non creata internamente.
    """

    def __init__(self, llm: LLM):
        """
        Args:
            llm: Istanza singleton di Filippo.LLM già configurata.
        """
        self._llm = llm

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """
        Genera testo usando il server LLM remoto.
        Wrappa la chiamata async in sync.
        
        Args:
            prompt: Il prompt da inviare all'LLM.
            max_tokens: Numero massimo di token da generare.
            temperature: Temperatura per la generazione.
        
        Returns:
            Il testo generato, oppure stringa di errore se fallisce.
        """
        try:
            # Wrappa chiamata async in sync
            return asyncio.run(
                self._llm.generate_text(prompt, max_tokens=max_tokens, temperature=temperature)
            )
        except Exception as e:
            # Fallback: ritorna errore leggibile per debug
            return f"[LLM error] {e}"

    def get_embedding(self, text: str) -> list[float]:
        """
        Usa il modello di embedding locale di Filippo.LLM.
        
        Args:
            text: Testo da cui estrarre l'embedding.
        
        Returns:
            Lista di float rappresentante l'embedding.
        """
        return self._llm.get_embedding(text)
