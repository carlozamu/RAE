import time
from typing import Callable


from Carlo.model.llm_client import LLMClient
from Edoardo.Genome.agent_genome import AgentGenome
from Edoardo.Traits.traits import Trait


class Phenotype:
    """
    Executable wrapper around an AgentGenome. Builds Traits from PromptNodes
    and executes them sequentially using the provided LLM client.
    
    Gestisce anche:
    - Ciclo di vita (age, alive, min_age)
    - Logging delle chiamate LLM per debug
    """

    def __init__(
        self,
        genome: AgentGenome,
        llm_client: LLMClient,
        template: str | None = None,
        min_age: int = 3,
    ):
        self.genome = genome
        self.llm = llm_client
        self.template = template
        self.traits = self._build_traits()

        # Ciclo di vita
        self.age: int = 0
        self.alive: bool = True
        self.min_age: int = min_age  # Iterazioni minime prima di poter essere eliminato

        # Debug logging: lista di {node_id, node_name, prompt, response, timestamp}
        self.call_log: list[dict] = []

    def _build_traits(self) -> list[Trait]:
        chain = self.genome.get_linear_chain()
        return [Trait(node, self.llm, self.template) for node in chain]

    # --- Ciclo di vita ---

    def step(self) -> None:
        """Incrementa l'età del fenotipo di 1 iterazione."""
        self.age += 1

    def can_be_eliminated(self) -> bool:
        """Verifica se il fenotipo può essere eliminato (ha superato età minima)."""
        return self.age >= self.min_age

    def kill(self) -> None:
        """Marca il fenotipo come morto."""
        self.alive = False

    # --- Evaluate (commentato per uso futuro) ---
    
    # def evaluate(
    #     self,
    #     fitness_fn: Callable[[str, str], float],
    #     target: str,
    #     initial_input: str = "",
    # ) -> float:
    #     """
    #     Esegue il phenotype e calcola la fitness.
    #     Salva il risultato in genome.fitness.
    #     
    #     Args:
    #         fitness_fn: Funzione che prende (output, target) e ritorna score.
    #         target: La risposta attesa.
    #         initial_input: Input iniziale per la catena.
    #     
    #     Returns:
    #         Lo score di fitness calcolato.
    #     """
    #     outputs = self.run(initial_input=initial_input)
    #     final_output = outputs[-1] if outputs else ""
    #     score = fitness_fn(final_output, target)
    #     self.genome.fitness = score
    #     return score

    # --- Esecuzione ---

    def run(self, initial_input: str = "", answer_only: bool = True) -> list[str]:
        """
        Execute the chain of traits. Each output becomes the next input.
        Returns the list of outputs from each node.
        Popola call_log con ogni chiamata per debug.
        """
        outputs = []
        current_input = initial_input
        for trait in self.traits:
            # Costruisce il prompt per logging
            prompt = trait.template.format(
                instruction=trait.node.instruction, input=current_input
            )
            
            current_output = trait.run(current_input)
            
            # Log della chiamata
            self.call_log.append({
                "node_id": trait.node.id,
                "node_name": trait.node.name,
                "prompt": prompt,
                "response": current_output,
                "timestamp": time.time(),
            })
            if answer_only:
                # Extract at most two bullet lines; fallback to first sentence of cleaned text.
                bullets = []
                for ln in current_output.splitlines():
                    stripped = ln.strip()
                    lower = stripped.lower()
                    if not stripped:
                        continue
                    if lower.startswith(("user", "input", "answer")):
                        continue
                    if stripped.startswith(("-", "*", "•")):
                        bullets.append(stripped)
                    if len(bullets) >= 2:
                        break
                if bullets:
                    cleaned = "\n".join(bullets[:2])
                    outputs.append(cleaned)
                    current_input = cleaned
                else:
                    cleaned_text = " ".join(
                        ln.strip()
                        for ln in current_output.splitlines()
                        if ln.strip() and not ln.strip().lower().startswith(("user", "input", "answer"))
                    )
                    first_sentence = cleaned_text.split(".")[0].strip() if cleaned_text else ""
                    outputs.append(first_sentence or cleaned_text)
                    current_input = first_sentence or cleaned_text
            else:
                outputs.append(current_output)
                current_input = current_output
        
        # Incrementa età dopo ogni run
        self.step()
        return outputs
