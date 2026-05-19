from math import inf
from typing import Dict, Any
from Gene.gene import PromptNode
from Genome.agent_genome import AgentGenome
from Traits.traits import Trait
from Utils.LLM import LLM

class Phenotype:
    def __init__(self, genome: AgentGenome, llm_client: LLM):
        self.genome = genome
        self.llm = llm_client

    async def run(self, problem: str) -> Dict[str, Any]:
        # 1. Get Plan
        # execution_order is [(NodeObject, [Parent_IDs]), ...]
        execution_order: list[tuple[PromptNode, list[int]]] = self.genome.get_execution_order() 
        
        # 2. Lightweight Memory (ID -> Instructions, Answer String)
        trait_answers: Dict[int, tuple[str, str]] = {}
        
        total_in_tokens = 0
        total_out_tokens = 0
        total_time = 0.0

        # 3. Execution Loop
        for node, parent_ids in execution_order:
            
            # A. Build Context
            context_parts = [f"<start_of_turn>system\n{problem}<end_of_turn>\n"]
            
            if parent_ids:
                for pid in parent_ids:
                    # We only fetch the ints, saving memory
                    if pid in trait_answers:
                        parent_instructions, parent_answer = trait_answers[pid]
                        parent_turn = f"<start_of_turn>user\n{parent_instructions}<end_of_turn>\n<start_of_turn>model\n{parent_answer}<end_of_turn>\n"
                        context_parts.append(parent_turn)
                     
            # Chek if it is the last node
            context_parts.append(f"<start_of_turn>user\n{node.instruction}<end_of_turn>\n<start_of_turn>model\n") # Leave model turn open for answer

            full_context = "".join(context_parts)
            
            # B. Execute (Transient)
            trait = Trait(node, self.llm)
            # Fix: renamed 'time' to 'duration' to avoid shadowing module
            last = False
            if node.id == self.genome.end_node_innovation_number:
                last= True
            in_t, out_t, duration, answer = await trait.execute(full_context, last)
            last = False
            
            # C. Store Result
            trait_answers[node.id] = (node.instruction, answer)
            
            # D. Accumulate Stats
            total_in_tokens += in_t
            total_out_tokens += out_t
            total_time += duration

        # 4. Final Result
        final_answer = trait_answers[execution_order[-1][0].id][1]
        
        final_object = {
            "answer": final_answer,
            "stats": {
                "total_tokens": total_in_tokens + total_out_tokens,
                "input_tokens": total_in_tokens,
                "output_tokens": total_out_tokens,
                "time_taken": total_time,
                "steps": len(execution_order)
            }
        }

        #print(f"\n\n\nPhenotype run completed. Answer: {final_object}\n\n\n")
        return final_object

    def deep_copy(self):
        return Phenotype(
            genome=self.genome.copy(),
            llm_client=self.llm
        )
