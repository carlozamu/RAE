from Filippo.Gene import PromptNode
from Carlo.model.llm_client import LLMClient


class Trait:
    """
    Wrapper around a PromptNode that knows how to execute it via an LLMClient.
    """

    def __init__(self, node: PromptNode, llm_client: LLMClient, template: str | None = None):
        self.node = node
        self.llm = llm_client
        # Simple default template if none provided.
        # We avoid the string "User" to keep LLM outputs cleaner for downstream nodes.
        self.template = template or "{instruction}\nInput: {input}\nAnswer:"

    def run(self, user_input: str) -> str:
        prompt = self.template.format(instruction=self.node.instruction, input=user_input)
        return self.llm.generate(prompt)
