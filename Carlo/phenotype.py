from Filippo.AgentGenome import AgentGenome
from Carlo.model.llm_client import LLMClient
from Carlo.trait import Trait


class Phenotype:
    """
    Executable wrapper around an AgentGenome. Builds Traits from PromptNodes
    and executes them sequentially using the provided LLM client.
    """

    def __init__(self, genome: AgentGenome, llm_client: LLMClient, template: str | None = None):
        self.genome = genome
        self.llm = llm_client
        self.template = template
        self.traits = self._build_traits()

    def _build_traits(self) -> list[Trait]:
        chain = self.genome.get_linear_chain()
        return [Trait(node, self.llm, self.template) for node in chain]

    def run(self, initial_input: str = "", answer_only: bool = True) -> list[str]:
        """
        Execute the chain of traits. Each output becomes the next input.
        Returns the list of outputs from each node.
        """
        outputs = []
        current_input = initial_input
        for trait in self.traits:
            current_output = trait.run(current_input)
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
        return outputs
