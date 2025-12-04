import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import time

class UnifiedFitnessCalculator:
    """
    Calcolatore di Fitness con Threshold che integra:
    - Risposta (Accuracy)
    - Ragionamento (Rationale)
    - Costo Token (Verbosità)
    - Complessità Ciclomatica del Grafo del Prompt (NEAT)
    """
    def __init__(self,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 w_accuracy=2.0,         # Peso per la correttezza della risposta (semantica)
                 w_rationale=2.0,        # Peso per la qualità del ragionamento (se presente)
                 w_token_cost=0.001,     # Penalità per la lunghezza (verbosità)
                 w_complexity_cost=0.07): # Penalità per la complessità del grafo del prompt
        
        print(f"Caricamento modello di embedding: {model_name}...")
        # Inizializzazione del modello di embedding
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        # Pesi della Fitness Function
        self.w_acc = w_accuracy
        self.w_rat = w_rationale
        self.w_tok = w_token_cost
        self.w_complexity = w_complexity_cost
        
        print("Modello caricato e calcolatore pronto.")

    def _calculate_semantic_similarity(self, text1, text2):
        """
        Calcola la Cosine Similarity tra due testi usando gli embeddings.
        Output: Float tra -1.0 (opposti) e 1.0 (identici).
        """
        # Gestione casi vuoti per evitare errori
        if not text1 or not text2:
            return 0.0
            
        # Creazione embeddings
        emb1 = self.embedding_model.embed_documents([text1])
        emb2 = self.embedding_model.embed_documents([text2])
        
        # Calcolo similarità coseno
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        return similarity

    def _calculate_cyclomatic_complexity(self, num_nodes, num_edges):
        """
        Calcola la Complessità Ciclomatica del grafo del prompt.
        Formula: CC = E - N + 2
        
        Dove:
        - E = numero di archi (edges/links)
        - N = numero di nodi (nodes)
        
        La CC misura il numero di percorsi indipendenti attraverso il grafo.
        Valori più alti indicano maggiore complessità e più branch decisions.
        
        Args:
            num_nodes: Numero di nodi nel grafo NEAT
            num_edges: Numero di archi nel grafo NEAT
            
        Returns:
            Complessità Ciclomatica (int)
        """
        if num_nodes <= 0:
            return 0
        
        # Formula classica della Complessità Ciclomatica
        cyclomatic_complexity = num_edges - num_nodes + 2
        
        # Assicuriamoci che non sia negativa (per grafi molto semplici)
        return max(0, cyclomatic_complexity)

    def compute(self, 
                generated_ans, target_ans, 
                generated_rat=None, target_rat=None, 
                num_nodes=1, num_edges=0,
                similarity_threshold=0.8):
        """
        Calcola la loss totale considerando:
        1. Correttezza della risposta (con threshold)
        2. Qualità del ragionamento (se presente)
        3. Costo dei token (verbosità)
        4. Complessità del grafo del prompt (Complessità Ciclomatica)
        
        Args:
            generated_ans: Risposta generata dal modello
            target_ans: Risposta target/corretta
            generated_rat: Ragionamento generato (opzionale)
            target_rat: Ragionamento target (opzionale)
            num_nodes: Numero di nodi nel grafo NEAT
            num_edges: Numero di archi nel grafo NEAT
            similarity_threshold: Soglia per considerare una risposta corretta
            
        Returns:
            Dict con 'loss' totale e 'details' dei componenti
        """
        
        details = {}
        
        # --- 1. Valutazione Risposta ---
        ans_similarity = self._calculate_semantic_similarity(generated_ans, target_ans)
        
        # Logica Threshold Risposta
        if ans_similarity >= similarity_threshold:
            ans_loss = 0.0
            is_ans_correct = True
        else:
            ans_loss = 1.0 - ans_similarity
            is_ans_correct = False
            
        weighted_ans_cost = ans_loss * self.w_acc
        
        details['ans_sim'] = round(ans_similarity, 4)
        details['ans_cost'] = round(weighted_ans_cost, 4)
        details['ans_ok'] = is_ans_correct

        # --- 2. Valutazione Rationale ---
        weighted_rat_cost = 0.0
        if generated_rat and target_rat:
            rat_similarity = self._calculate_semantic_similarity(generated_rat, target_rat)
            
            # Soglia più bassa per il Reasoning (0.8 - 0.2 = 0.6)
            rat_threshold = similarity_threshold - 0.2 
            
            # Logica Threshold Rationale
            if rat_similarity >= rat_threshold:
                rat_loss = 0.0
                is_rat_correct = True
            else:
                rat_loss = 1.0 - rat_similarity
                is_rat_correct = False
            
            weighted_rat_cost = rat_loss * self.w_rat
            
            details['rat_sim'] = round(rat_similarity, 4)
            details['rat_cost'] = round(weighted_rat_cost, 4)
            details['rat_ok'] = is_rat_correct
        
        # --- 3. Costo Token (Verbosità) ---
        total_text = generated_ans + (" " + generated_rat if generated_rat else "")
        token_count = len(total_text.split())
        
        token_cost = token_count * self.w_tok
        
        details['token_count'] = token_count
        details['token_cost'] = round(token_cost, 4)

        # --- 4. Costo Complessità Ciclomatica ---
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(num_nodes, num_edges)
        complexity_cost = cyclomatic_complexity * self.w_complexity
        
        details['num_nodes'] = num_nodes
        details['num_edges'] = num_edges
        details['cyclomatic_complexity'] = cyclomatic_complexity
        details['complexity_cost'] = round(complexity_cost, 4)

        # --- Calcolo Loss Totale ---
        total_loss = weighted_ans_cost + weighted_rat_cost + token_cost + complexity_cost
        
        return {
            "loss": round(total_loss, 4),
            "details": details
        }

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    # Inizializzazione
    calculator = UnifiedFitnessCalculator(
        w_accuracy=10.0,        # Priorità alta alla risposta corretta
        w_rationale=5.0,        # Priorità media al ragionamento
        w_token_cost=0.01,  
        w_complexity_cost=0.5   # Penalità per grafi complessi
    )
    
    # Caso 1: Grafo Semplice (Lineare)
    print("\n--- TEST 1: Grafo Lineare Semplice ---")
    gen_a = "The vehicle is moving fast"
    ref_a = "The car is traveling quickly"
    
    res1 = calculator.compute(
        gen_a, ref_a, 
        num_nodes=3, num_edges=2,  # Grafo lineare: A->B->C
        similarity_threshold=0.8
    )
    print(f"Gen: '{gen_a}' vs Ref: '{ref_a}'")
    print(f"Grafo: 3 nodi, 2 archi -> CC = {res1['details']['cyclomatic_complexity']}")
    print(f"Loss: {res1['loss']}")
    print(f"Details: {res1['details']}")
    
    # Caso 2: Grafo Complesso con Branch
    print("\n--- TEST 2: Grafo con Branch (più complesso) ---")
    gen_rat = "Because the sun is a star, it emits light."
    ref_rat = "Since the sun is classified as a star, it produces light energy."
    
    res2 = calculator.compute(
        generated_ans=gen_a, 
        target_ans=ref_a,
        generated_rat=gen_rat,
        target_rat=ref_rat,
        num_nodes=5, num_edges=7  # Grafo con cicli e branch
    )
    print(f"Grafo: 5 nodi, 7 archi -> CC = {res2['details']['cyclomatic_complexity']}")
    print(f"Loss: {res2['loss']}")
    print(f"Details: {res2['details']}")
    
    # Caso 3: Grafo Molto Complesso
    print("\n--- TEST 3: Grafo Molto Complesso ---")
    res3 = calculator.compute(
        generated_ans=gen_a, 
        target_ans=ref_a,
        num_nodes=10, num_edges=20  # Grafo molto complesso
    )
    print(f"Grafo: 10 nodi, 20 archi -> CC = {res3['details']['cyclomatic_complexity']}")
    print(f"Loss: {res3['loss']}")
    print(f"Details: {res3['details']}")