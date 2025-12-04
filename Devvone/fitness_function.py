import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import time

class UnifiedFitnessCalculator: # Calcolatore di Fitness con Threshold che integra Risposta, Ragionamento, Costo Token e Tempo (Unified -> Ans e/o Rationale)
    def __init__(self,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 w_accuracy=2.0,       # Peso per la correttezza della risposta (semantica)
                 w_rationale=2.0,      # Peso per la qualità del ragionamento (se presente)
                 w_token_cost=0.001,   # Penalità per la lunghezza (verbosità)
                 w_time_cost=0.1):     # Penalità per il tempo di esecuzione
        
        print(f"Caricamento modello di embedding: {model_name}...")
        # Inizializzazione del modello di embedding
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        # Pesi della Fitness Function
        self.w_acc = w_accuracy
        self.w_rat = w_rationale
        self.w_tok = w_token_cost
        self.w_time = w_time_cost
        
        print("Modello caricato e calcolatore pronto.")

    def _calculate_semantic_similarity(self, text1, text2):
        """
        Calcola la Cosine Similarity tra due testi usando gli embeddings.
        Output: Float tra -1.0 (opposti) e 1.0 (identici).
        """
        # Gestione casi vuoti per evitare errori
        if not text1 or not text2:
            return 0.0
            
        # Creazione embeddings (embed_documents accetta una lista di stringhe)
        # Nota: usiamo [text] perché la funzione si aspetta una lista
        emb1 = self.embedding_model.embed_documents([text1])
        emb2 = self.embedding_model.embed_documents([text2])
        
        # Calcolo similarità coseno
        # cosine_similarity restituisce una matrice [[score]], prendiamo [0][0]
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        return similarity

    def compute(self, 
                generated_ans, target_ans, 
                generated_rat=None, target_rat=None, 
                time_taken=0.0,
                similarity_threshold=0.8): # Soglia principale per la Risposta (Ans)
        
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
        
        # --- 3. Costi Risorse ---
        total_text = generated_ans + (" " + generated_rat if generated_rat else "")
        token_count = len(total_text.split())
        
        token_cost = token_count * self.w_tok
        time_cost = time_taken * self.w_time
        
        details['token_cost'] = round(token_cost, 4)
        details['time_cost'] = round(time_cost, 4)

        # --- Calcolo Totale ---
        total_loss = weighted_ans_cost + weighted_rat_cost + token_cost + time_cost
        
        return {
            "loss": round(total_loss, 4),
            "details": details
        }

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    # Inizializzazione
    calculator = UnifiedFitnessCalculator(
        w_accuracy=10.0,    # Priorità alta alla risposta corretta
        w_rationale=5.0,    # Priorità media al ragionamento
        w_token_cost=0.01,  
        w_time_cost=0.1
    )
    
    # Caso 1: Solo Risposta (Semanticamente simile ma parole diverse)
    print("\n--- TEST 1: Solo Risposta ---")
    gen_a = "The vehicle is moving fast"
    ref_a = "The car is traveling quickly" # Stesso significato, parole diverse
    
    res1 = calculator.compute(gen_a, ref_a, time_taken=0.5)
    print(f"Gen: '{gen_a}' vs Ref: '{ref_a}'")
    print(f"Loss: {res1['loss']}")
    print(f"Details: {res1['details']}")
    
    # Caso 2: Risposta + Rationale
    print("\n--- TEST 2: Risposta + Rationale ---")
    gen_rat = "Because the sun is a star, it emits light."
    ref_rat = "Since the sun is classified as a star, it produces light energy."
    
    res2 = calculator.compute(
        generated_ans=gen_a, 
        target_ans=ref_a,
        generated_rat=gen_rat,
        target_rat=ref_rat,
        time_taken=1.2
    )
    print(f"Gen Rat: '{gen_rat}' \nRef Rat: '{ref_rat}'")
    print(f"Loss: {res2['loss']}")
    print(f"Details: {res2['details']}")