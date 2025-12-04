import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import time

from fitness_function import UnifiedFitnessCalculator

# --- 2. SUITE DI TEST ---
if __name__ == "__main__":
    
    # Setup del calcolatore
    calculator = UnifiedFitnessCalculator(
        w_accuracy=10.0,   # Alta penalità se sbaglia
        w_rationale=5.0,   # Media penalità se sbaglia ragionamento
        w_token_cost=0.01,
        w_time_cost=0.0
    )

    # Definizione dei casi di test
    test_cases = [
        {
            "id": "1. Perfect Match",
            "desc": "Risposta identica (Dovrebbe avere Loss ~0 + costo token)",
            "gen_ans": "Paris",
            "ref_ans": "Paris",
            "gen_rat": None, "ref_rat": None
        },
        {
            "id": "2. Semantic Match (Threshold Test)",
            "desc": "Parole diverse, stesso senso. La similarity deve essere > 0.8 per avere Loss 0.",
            "gen_ans": "The car is very fast",
            "ref_ans": "The automobile moves at high speed",
            "gen_rat": None, "ref_rat": None
        },
        {
            "id": "3. Wrong Answer",
            "desc": "Risposta sbagliata. La similarity deve essere bassa e la Loss alta.",
            "gen_ans": "Berlin",
            "ref_ans": "Paris",
            "gen_rat": None, "ref_rat": None
        },
        {
            "id": "4. Math Reasoning (Correct)",
            "desc": "Risposta e Ragionamento corretti ma formulati diversamente.",
            "gen_ans": "4",
            "ref_ans": "four",
            "gen_rat": "2 plus 2 equals 4.",
            "ref_rat": "The sum of two and two is four."
        },
        {
            "id": "5. Math Reasoning (Bad Rationale)",
            "desc": "Risposta giusta, ma ragionamento errato/confuso.",
            "gen_ans": "4",
            "ref_ans": "4",
            "gen_rat": "I guessed the number randomly.",
            "ref_rat": "Since 2+2 is a basic arithmetic operation resulting in 4."
        },
        {
            "id": "6. Complex Context (Bio)",
            "desc": "Test su frase lunga scientifica.",
            "gen_ans": "Mitochondria",
            "ref_ans": "The mitochondria",
            "gen_rat": "It is the powerhouse of the cell responsible for energy.",
            "ref_rat": "Mitochondria generate most of the chemical energy needed to power the cell's biochemical reactions."
        },
        {
            "id": "7. Ambiguous/Vague (Edge Case)",
            "desc": "Risposta vaga. Riuscirà a superare la soglia 0.8?",
            "gen_ans": "Maybe it is a vehicle",
            "ref_ans": "It is a car",
            "gen_rat": None, "ref_rat": None
        }
    ]

    print(f"{'ID':<30} | {'Sim Ans':<10} | {'Sim Rat':<10} | {'LOSS TOT':<10} | {'Esito'}")
    print("-" * 90)

    for case in test_cases:
        res = calculator.compute(
            generated_ans=case["gen_ans"],
            target_ans=case["ref_ans"],
            generated_rat=case["gen_rat"],
            target_rat=case["ref_rat"],
            time_taken=0.5,
            similarity_threshold=0.8  # <--- SOGLIA IMPOSTATA QUI
        )
        
        details = res['details']
        
        # Formattazione output per tabella
        rat_sim_str = str(details.get('rat_sim', 'N/A'))
        ans_ok = "OK" if details['ans_ok'] else "FAIL"
        rat_ok = "OK" if details.get('rat_ok', False) else "FAIL" if case["gen_rat"] else "-"
        
        print(f"{case['id']:<30} | {details['ans_sim']:<10} | {rat_sim_str:<10} | {res['loss']:<10} | Ans:{ans_ok} Rat:{rat_ok}")
        # Se vuoi vedere i dettagli completi scommenta la riga sotto
        # print(f"   -> Desc: {case['desc']}") 
        # print("-" * 90)