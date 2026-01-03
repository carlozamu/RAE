from Edoardo.Fitness.fitness_function import UnifiedFitnessCalculator
from Edoardo.Utils.LLM import LLM

def run_analysis():
    print("Initializing LLM for baseline analysis...")
    llm = LLM()
    
    # Initialize Calculator with standard weights
    # w_accuracy=2.0, w_rationale=2.0, w_token_cost=0.001, w_complexity_cost=0.07,
    calc = UnifiedFitnessCalculator(llm=llm, w_accuracy=2.0, w_rationale=2.0, w_token_cost=0.001, w_complexity_cost=0.07)
    
    print("\n--- Baseline Loss Analysis ---")
    
    # Case 1: Perfect Match (Cluttr-style, single word)
    # 0 Semantic Loss, Minimal Token Cost, Low Complexity
    print("\n[Case 1] Perfect Single Word (e.g. 'aunt')")
    res1 = calc.compute(
        generated_ans="aunt", target_ans="aunt",
        num_nodes=3, num_edges=2 
    )
    print(f"Loss: {res1['loss']} | Details: {res1['details']}")
    
    # Case 2: Perfect Match with Rationale (CoT-style)
    # 0 Semantic Loss, 0 Rationale Loss, Moderate Tokens
    print("\n[Case 2] Perfect CoT (Ans + Rat)")
    gen_ans = "The answer is 4."
    gen_rat = "2 plus 2 is 4."
    res2 = calc.compute(
        generated_ans=gen_ans, target_ans=gen_ans,
        generated_rat=gen_rat, target_rat=gen_rat,
        num_nodes=5, num_edges=5
    )
    print(f"Loss: {res2['loss']} | Details: {res2['details']}")
    
    # Case 3: Good Answer, slightly different wording
    print("\n[Case 3] Good Answer, imperfect wording")
    res3 = calc.compute(
        generated_ans="It is 4", target_ans="The answer is 4",
        num_nodes=5, num_edges=5
    )
    print(f"Loss: {res3['loss']} | Details: {res3['details']}")
    
    # Case 4: Wrong Answer
    print("\n[Case 4] Wrong Answer")
    res4 = calc.compute(
        generated_ans="Blue", target_ans="Red",
        num_nodes=5, num_edges=5
    )
    print(f"Loss: {res4['loss']} | Details: {res4['details']}")

    print("\n--- Recommendation ---")
    print("Set TARGET_FITNESS slightly above the Perfect Case baseline.")

if __name__ == "__main__":
    run_analysis()
