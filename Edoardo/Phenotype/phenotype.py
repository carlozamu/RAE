from math import inf
from typing import Dict, Any
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
        execution_order = self.genome.get_execution_order() 
        
        # 2. Lightweight Memory (ID -> Answer String)
        trait_answers: Dict[str, str] = {}
        
        total_in_tokens = 0
        total_out_tokens = 0
        total_time = 0.0

        # 3. Execution Loop
        for node, parent_ids in execution_order:
            
            # A. Build Context
            context_parts = [f"Problem:\n{problem}\n"]
            
            if parent_ids:
                context_parts.append("Reasoning Context:")
                for pid in parent_ids:
                    # We only fetch the string, saving memory
                    if pid in trait_answers:
                        # OPTIONAL: You might want to prefix this with node.name 
                        # if you have access to the parent node object easily.
                        # For now, raw answer is efficient.
                        context_parts.append(trait_answers[pid])
            
            full_context = "\n".join(context_parts)
            
            # B. Execute (Transient)
            trait = Trait(node, self.llm)
            # Fix: renamed 'time' to 'duration' to avoid shadowing module
            in_t, out_t, duration, answer = await trait.execute(full_context)
            
            # C. Store Result
            trait_answers[node.id] = answer
            
            # D. Accumulate Stats
            total_in_tokens += in_t
            total_out_tokens += out_t
            total_time += duration

        # 4. Final Result
        final_answer = trait_answers[execution_order[-1][0].id]
        
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
