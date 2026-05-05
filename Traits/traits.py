import time
from Utils.LLM import LLM
from Gene.gene import PromptNode

class Trait:
    """
    Transient executor for a single gene.
    """
    def __init__(self, node: PromptNode, llm_client: LLM, primer: str = ""):
        self.node = node
        self.llm = llm_client
        self.primer = primer

    async def execute(self, context: str) -> tuple[int, int, float, str]:
        """
        Executes and returns (in_tokens, out_tokens, duration, answer).
        """
        user_prompt =f"<start_of_turn>user\n{self.node.instruction}<end_of_turn>\n<start_of_turn>model\n{self.primer}"
        prompt = context + user_prompt
        
        start_t = time.time()
        answer = await self.llm.generate_text(user_prompt=prompt)
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
    