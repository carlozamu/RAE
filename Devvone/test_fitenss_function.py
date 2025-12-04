import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import time

from fitness_function import UnifiedFitnessCalculator

# --- SUITE DI TEST COMPLETA ---
if __name__ == "__main__":
    
    # Setup del calcolatore
    calculator = UnifiedFitnessCalculator()

    # ========================================
    # PARTE 1: TEST ORIGINALI (Risposta + Ragionamento)
    # ========================================
    print("=" * 100)
    print("PARTE 1: TEST SEMANTICI (Risposta e Ragionamento)")
    print("=" * 100)
    
    test_cases_semantic = [
        {
            "id": "1. Perfect Match",
            "desc": "Risposta identica (Loss ~0 + costi)",
            "gen_ans": "Paris",
            "ref_ans": "Paris",
            "gen_rat": None, "ref_rat": None,
            "nodes": 2, "edges": 1  # Grafo minimo
        },
        {
            "id": "2. Semantic Match",
            "desc": "Parole diverse, stesso senso (sim > 0.8)",
            "gen_ans": "The car is very fast",
            "ref_ans": "The automobile moves at high speed",
            "gen_rat": None, "ref_rat": None,
            "nodes": 3, "edges": 2  # Grafo lineare semplice
        },
        {
            "id": "3. Wrong Answer",
            "desc": "Risposta sbagliata (alta Loss)",
            "gen_ans": "Berlin",
            "ref_ans": "Paris",
            "gen_rat": None, "ref_rat": None,
            "nodes": 3, "edges": 2
        },
        {
            "id": "4. Math (Correct)",
            "desc": "Risposta e Ragionamento corretti",
            "gen_ans": "4",
            "ref_ans": "four",
            "gen_rat": "2 plus 2 equals 4.",
            "ref_rat": "The sum of two and two is four.",
            "nodes": 4, "edges": 4  # Grafo con un branch
        },
        {
            "id": "5. Math (Bad Rationale)",
            "desc": "Risposta giusta, ragionamento errato",
            "gen_ans": "4",
            "ref_ans": "4",
            "gen_rat": "I guessed the number randomly.",
            "ref_rat": "Since 2+2 is a basic arithmetic operation resulting in 4.",
            "nodes": 4, "edges": 4
        },
        {
            "id": "6. Complex Context (Bio)",
            "desc": "Frase scientifica lunga",
            "gen_ans": "Mitochondria",
            "ref_ans": "The mitochondria",
            "gen_rat": "It is the powerhouse of the cell responsible for energy.",
            "ref_rat": "Mitochondria generate most of the chemical energy needed to power the cell's biochemical reactions.",
            "nodes": 5, "edges": 5
        },
        {
            "id": "7. Ambiguous/Vague",
            "desc": "Risposta vaga (edge case)",
            "gen_ans": "Maybe it is a vehicle",
            "ref_ans": "It is a car",
            "gen_rat": None, "ref_rat": None,
            "nodes": 3, "edges": 3
        }
    ]

    print(f"{'ID':<30} | {'Sim Ans':<8} | {'Sim Rat':<8} | {'CC':<4} | {'LOSS':<8} | {'Esito'}")
    print("-" * 100)

    for case in test_cases_semantic:
        res = calculator.compute(
            generated_ans=case["gen_ans"],
            target_ans=case["ref_ans"],
            generated_rat=case["gen_rat"],
            target_rat=case["ref_rat"],
            num_nodes=case["nodes"],
            num_edges=case["edges"],
            similarity_threshold=0.8
        )
        
        details = res['details']
        
        rat_sim_str = str(details.get('rat_sim', 'N/A'))
        ans_ok = "OK" if details['ans_ok'] else "FAIL"
        rat_ok = "OK" if details.get('rat_ok', False) else "FAIL" if case["gen_rat"] else "-"
        
        print(f"{case['id']:<30} | {details['ans_sim']:<8.4f} | {rat_sim_str:<8} | "
              f"{details['cyclomatic_complexity']:<4} | {res['loss']:<8.4f} | "
              f"Ans:{ans_ok} Rat:{rat_ok}")

    # ========================================
    # PARTE 2: TEST COMPLESSITÀ CICLOMATICA
    # ========================================
    print("\n" + "=" * 100)
    print("PARTE 2: TEST COMPLESSITÀ CICLOMATICA (Stesso Output, Grafi Diversi)")
    print("=" * 100)
    print("\nScenario: Tutti danno la stessa risposta corretta, ma con grafi di complessità crescente")
    print("Obiettivo: Verificare che grafi più complessi vengano penalizzati di più\n")
    
    test_cases_complexity = [
        {
            "id": "A. Grafo Minimo",
            "desc": "Grafo con 1 nodo, 0 archi (prompt semplicissimo)",
            "nodes": 1, "edges": 0,
            "expected_cc": 1  # CC = 0 - 1 + 2 = 1
        },
        {
            "id": "B. Grafo Lineare",
            "desc": "Catena semplice: A->B->C",
            "nodes": 3, "edges": 2,
            "expected_cc": 1  # CC = 2 - 3 + 2 = 1
        },
        {
            "id": "C. Grafo con 1 Branch",
            "desc": "Un'unica decisione if-then",
            "nodes": 4, "edges": 4,
            "expected_cc": 2  # CC = 4 - 4 + 2 = 2
        },
        {
            "id": "D. Grafo con 2 Branch",
            "desc": "Due decisioni if-then",
            "nodes": 6, "edges": 7,
            "expected_cc": 3  # CC = 7 - 6 + 2 = 3
        },
        {
            "id": "E. Grafo Complesso",
            "desc": "Molti branch e cicli",
            "nodes": 8, "edges": 12,
            "expected_cc": 6  # CC = 12 - 8 + 2 = 6
        },
        {
            "id": "F. Grafo Molto Complesso",
            "desc": "Prompt con logica molto articolata",
            "nodes": 10, "edges": 18,
            "expected_cc": 10  # CC = 18 - 10 + 2 = 10
        },
        {
            "id": "G. Grafo Estremamente Complesso",
            "desc": "Prompt over-engineered",
            "nodes": 15, "edges": 30,
            "expected_cc": 17  # CC = 30 - 15 + 2 = 17
        }
    ]

    # Usiamo sempre la stessa risposta corretta per isolare l'effetto della complessità
    correct_ans = "The capital of France is Paris"
    correct_rat = "Paris has been the capital of France since 987 AD."

    print(f"{'ID':<30} | {'Nodes':<6} | {'Edges':<6} | {'CC':<6} | {'CC Cost':<8} | {'LOSS Tot':<10}")
    print("-" * 100)

    for case in test_cases_complexity:
        res = calculator.compute(
            generated_ans=correct_ans,
            target_ans=correct_ans,  # Risposta perfetta
            generated_rat=correct_rat,
            target_rat=correct_rat,  # Ragionamento perfetto
            num_nodes=case["nodes"],
            num_edges=case["edges"],
            similarity_threshold=0.8
        )
        
        details = res['details']
        cc = details['cyclomatic_complexity']
        cc_cost = details['complexity_cost']
        
        # Verifica che la CC calcolata corrisponda a quella attesa
        status = "✓" if cc == case['expected_cc'] else f"✗ (atteso {case['expected_cc']})"
        
        print(f"{case['id']:<30} | {case['nodes']:<6} | {case['edges']:<6} | "
              f"{cc:<6} | {cc_cost:<8.4f} | {res['loss']:<10.4f} {status}")

    # ========================================
    # PARTE 3: TEST COMPARATIVI
    # ========================================
    print("\n" + "=" * 100)
    print("PARTE 3: CONFRONTO DIRETTO - Stessa Accuracy, Complessità Diversa")
    print("=" * 100)
    print("\nConfronto tra due prompt con stessa performance ma complessità diversa:\n")
    
    # Prompt A: Semplice ma efficace
    res_simple = calculator.compute(
        generated_ans="42",
        target_ans="42",
        generated_rat="It's the answer to everything.",
        target_rat="It's the answer to everything.",
        num_nodes=3,
        num_edges=2,
        similarity_threshold=0.8
    )
    
    # Prompt B: Complesso con stessa performance
    res_complex = calculator.compute(
        generated_ans="42",
        target_ans="42",
        generated_rat="It's the answer to everything.",
        target_rat="It's the answer to everything.",
        num_nodes=12,
        num_edges=20,
        similarity_threshold=0.8
    )
    
    print("PROMPT A (Semplice):")
    print(f"  - Nodi: 3, Archi: 2")
    print(f"  - CC: {res_simple['details']['cyclomatic_complexity']}")
    print(f"  - Loss Totale: {res_simple['loss']}")
    print(f"  - Breakdown: Ans={res_simple['details']['ans_cost']}, "
          f"Rat={res_simple['details'].get('rat_cost', 0)}, "
          f"Token={res_simple['details']['token_cost']}, "
          f"Complexity={res_simple['details']['complexity_cost']}")
    
    print("\nPROMPT B (Complesso):")
    print(f"  - Nodi: 12, Archi: 20")
    print(f"  - CC: {res_complex['details']['cyclomatic_complexity']}")
    print(f"  - Loss Totale: {res_complex['loss']}")
    print(f"  - Breakdown: Ans={res_complex['details']['ans_cost']}, "
          f"Rat={res_complex['details'].get('rat_cost', 0)}, "
          f"Token={res_complex['details']['token_cost']}, "
          f"Complexity={res_complex['details']['complexity_cost']}")
    
    diff = res_complex['loss'] - res_simple['loss']
    print(f"\n→ Differenza Loss: {diff:.4f}")
    print(f"→ Il prompt complesso è penalizzato di {diff:.4f} punti in più!")
    
    print("\n" + "=" * 100)
    print("CONCLUSIONE: La Complessità Ciclomatica penalizza efficacemente i grafi più complessi,")
    print("favorendo soluzioni più semplici a parità di performance.")
    print("=" * 100)