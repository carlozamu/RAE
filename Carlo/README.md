# Carlo's sandbox

Questa cartella contiene un wrapper minimale per eseguire un `AgentGenome` con un LLM stub o con Gemma 3, senza toccare il codice degli altri.

## Componenti
- `model/llm_client.py`: interfaccia `LLMClient` stub (echo), sostituibile con un client reale.
- `model/llm_gemma.py`: client HF per Gemma 3 270M (normale).
- `trait.py`: wrapper di `PromptNode` che costruisce il prompt da un template e chiama l'LLM.
- `phenotype.py`: costruisce i `Trait` da un `AgentGenome` e li esegue in sequenza (output → input del nodo successivo).
- `test/example.py`: esempio end-to-end con due nodi (riassunto → traduzione) usando lo stub.
- `test/example_gemma.py`: esempio con Gemma 3 al posto dello stub.

## Come provare
```bash
python Carlo/test/example.py
```

Output atteso: echo dello stub `[LLM stub] ...`.

Per usare Gemma 3 (serve modello via HF):
```bash
python Carlo/test/example_gemma.py
```

## Note
- Importa le classi esistenti (`Filippo.AgentGenome`, `Filippo.Gene`) senza modificarle.
- Sostituisci `LLMClient.generate` con una chiamata reale quando il modello è deciso (OpenAI/transformers/locale).
- Per l'ambiente: crea un venv (`python -m venv .venv && source .venv/bin/activate` su Unix) e installa le dipendenze da `requirements.txt` in root. 
- Per Gemma: il repo è gated, serve un token HF con permesso di leggere repo pubblici gated (`hf auth login`); il client ora evita `device_map="auto"` (niente `accelerate` richiesta). Se vuoi usare `device_map="auto"`, passa `use_device_map=True` a `Gemma3Client` e installa `accelerate`. 
