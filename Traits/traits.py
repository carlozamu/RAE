import time
from Utils.LLM import LLM
from Gene.gene import PromptNode

class Trait:
    """
    Transient executor for a single gene.
    """
    def __init__(self, node: PromptNode, llm_client: LLM):
        self.node = node
        self.llm = llm_client

    async def execute(self, context: str) -> tuple[int, int, float, str]:
        """
        Executes and returns (in_tokens, out_tokens, duration, answer).
        """
        prompt = f"""{context}\nInstruction: {self.node.instruction}\n"""
        
        start_t = time.time()
        answer = await self.llm.generate_text(user_prompt=prompt, primer="Answer: ")
        duration = time.time() - start_t
        
        in_tokens = len(prompt) // 4
        out_tokens = len(answer) // 4

        """ print(f"Executed Trait '{self.node.name}' (ID: {self.node.id}) in {duration:.2f}s | In Tokens: {in_tokens} | Out Tokens: {out_tokens}")
        print("--" * 20)
        print(f"Context: {context}")
        print("\n\n")
        print(f"Answer: {answer}")
        print("\n\n")
        print("--" * 20) """
        
        return in_tokens, out_tokens, duration, answer
    