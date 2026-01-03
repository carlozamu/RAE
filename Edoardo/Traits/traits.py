from Edoardo.Utils.LLM import LLM
from Edoardo.Gene.gene import PromptNode


class Trait:
    """
    Wrapper around a PromptNode that knows how to execute it via an LLM client.
    """

    def __init__(self, node: PromptNode, llm_client: LLM, context: str):
        self.node = node
        self.llm = llm_client
        self.context = context
        self.answer = self.get_response()
        self.in_tokens = (len(self.node.instruction) + len(self.context)) // 4  # Approximate token count
        self.out_tokens = len(self.answer) // 4  # Approximate token count

    def get_response(self) -> str:
        prompt = f"""Context: {self.context}\nInstruction: {self.node.instruction}\n"""
        return self.llm.generate_text(user_prompt=prompt, primer="Answer: ")
