# Carlo's sandbox

Questa cartella contiene un wrapper minimale per eseguire un `AgentGenome` con un LLM stub o con Gemma 3, senza toccare il codice degli altri.

## Componenti

### Model Layer
- `model/llm_client.py`: interfaccia base `LLMClient` stub (echo), sostituibile con un client reale.
- `model/llm_gemma.py`: client HF per Gemma 3 270M (locale).
- `model/llm_remote.py`: **[NEW]** adapter per il server LLM remoto di Filippo (`Filippo.LLM`).
- `model/llm_mock.py`: **[NEW]** mock configurabile per test senza server.

### Core
- `trait.py`: wrapper di `PromptNode` che costruisce il prompt da un template e chiama l'LLM.
- `phenotype.py`: costruisce i `Trait` da un `AgentGenome` e li esegue in sequenza.
  - **[NEW]** Gestisce ciclo di vita: `age`, `alive`, `min_age`, `can_be_eliminated()`, `kill()`
  - **[NEW]** Logging debug: `call_log` con ogni chiamata LLM

### Test
- `test/example.py`: esempio end-to-end con stub.
- `test/example_gemma.py`: esempio con Gemma 3 locale.
- `test/example_remote.py`: **[NEW]** esempio con server LLM remoto.
- `test/example_with_fitness.py`: esempio con calcolo fitness.

## Come provare

### Con stub (nessun modello richiesto)
```bash
python Carlo/test/example.py
```

### Con server LLM remoto (richiede server attivo)
```bash
# Prima avvia il server (es. vLLM su localhost:8000)
python Carlo/test/example_remote.py
```

### Con Gemma 3 locale (richiede HF)
```bash
python Carlo/test/example_gemma.py
```

## Nuove funzionalità

### Ciclo di vita del Phenotype
```python
phenotype = Phenotype(genome, llm_client, min_age=3)

phenotype.run(...)  # età incrementa automaticamente
print(phenotype.age)  # 1

if phenotype.can_be_eliminated():
    phenotype.kill()
```

### Debug logging
```python
phenotype.run(initial_input="...")

for entry in phenotype.call_log:
    print(f"[{entry['node_name']}] {entry['response'][:50]}...")
```

## Note
- Importa le classi esistenti (`Filippo.AgentGenome`, `Filippo.Gene`, `Filippo.LLM`) senza modificarle.
- Il metodo `evaluate()` è commentato nel codice, pronto per uso futuro.
- Per l'ambiente: crea un venv (`python -m venv .venv && source .venv/bin/activate` su Unix) e installa le dipendenze da `requirements.txt` in root.
