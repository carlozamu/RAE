import os
import numpy as np
from Utils.LLM import LLM

MODEL_NAME = "google/gemma-3-1b-it" 
BASE_URL = "http://localhost:8000"  

# ==========================================
# 1. THE 30 TEST PROMPTS
# ==========================================
PROMPTS = [
    # Group A: Kinship Extraction
    "Identify the kinship term from the provided options.",
    "Select the kinship term from the options provided.",
    "Determine the correct kinship term.",
    "Choose the appropriate kinship word.",
    "What is the kinship relationship?",
    
    # Group B: Step-by-Step Reasoning
    "Think step-by-step to reach a logical conclusion.",
    "Let's think step by step.",
    "Break down the problem step by step.",
    "Analyze the situation sequentially, step by step.",
    "Provide a step-by-step logical deduction.",
    
    # Group C: Strict Single-Word Formatting
    "Output only a single word.",
    "Provide exactly one word as the answer.",
    "State only the single word.",
    "Your answer must be one word only.",
    "Constraint: output one word.",
    
    # Group D: General Logical Deduction
    "Deduce the hidden connection between the entities.",
    "Find the logical link between the subjects.",
    "Identify how the two people are connected.",
    "Determine the exact relational link.",
    "Trace the logical connection.",
    
    # Group E: Total Noise / Irrelevant
    "Calculate the trajectory of the projectile.",
    "Solve the differential equation.",
    "Explain the theory of general relativity.",
    "Summarize the rules of quantum mechanics.",
    "Compute the derivative of the function.",
    
    # Group F: Anti-Yapping Formatting
    "Do not write a full sentence.",
    "No conversational filler allowed.",
    "Do not provide conversational filler or explanations.",
    "Output the answer without any extra text.",
    "Omit all conversational text and output the raw answer."
]

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_flat = np.array(v1).flatten()
    v2_flat = np.array(v2).flatten()
    return float(np.dot(v1_flat, v2_flat) / (np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat)))

# ==========================================
# 2. MAIN TEST & LOGGING FUNCTION
# ==========================================
def run_exhaustive_test():
    print("Initializing LLM to fetch embeddings...")
    llm = LLM(model_name=MODEL_NAME, base_url=BASE_URL) 
    
    print("\nEmbedding all 30 prompts...")
    embeddings = []
    for i, p in enumerate(PROMPTS):
        emb = llm.get_embedding(p)
        embeddings.append(emb)
        print(f"Embedded [{i+1}/30]")
        
    print("\nCalculating exhaustive pairwise distances...")
    
    md_lines = [
        "# 🧠 Exhaustive Pairwise Semantic Similarity Matrix",
        "This report shows the exact cosine similarity between every prompt and all other prompts in the dataset.",
        "Look at the top 3-4 matches for each prompt to determine the optimal `similarity_threshold` for your Semantic Registry.\n",
        "---\n"
    ]
    
    for i, target_prompt in enumerate(PROMPTS):
        md_lines.append(f"### 🎯 Prompt {i+1}: `{target_prompt}`")
        md_lines.append("| Rank | Similarity | Compared Prompt |")
        md_lines.append("| :---: | :---: | :--- |")
        
        comparisons = []
        for j, comp_prompt in enumerate(PROMPTS):
            if i == j:
                continue # Skip comparing the prompt to itself
            sim = cosine_similarity(embeddings[i], embeddings[j])
            comparisons.append((sim, comp_prompt))
            
        # Sort by most similar first
        comparisons.sort(key=lambda x: x[0], reverse=True)
        
        for rank, (sim, txt) in enumerate(comparisons):
            # Highlight high similarities in bold for easy scanning
            sim_str = f"**{sim:.4f}**" if sim > 0.80 else f"{sim:.4f}"
            md_lines.append(f"| {rank+1} | {sim_str} | `{txt}` |")
            
        md_lines.append("\n<br>\n")

    log_file = "registry_test.md"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"\n✅ Matrix Complete! Report saved to: {os.path.abspath(log_file)}")

if __name__ == "__main__":
    run_exhaustive_test()