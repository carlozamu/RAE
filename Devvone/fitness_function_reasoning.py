import evaluate
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class CostCalculatorSBERT:
    def __init__(self,
                 w_accuracy=2.0,       # Absolute priority to answer accuracy
                 w_rationale=2.0,      # Highe weight to rationale quality
                 w_token_cost=0.05,    # Cost for verbosity
                 w_time_cost=0.5,      # Cost for latency
                 w_rouge_ans=1.0):     # Similarity to brief answer structure

        self.w_acc = w_accuracy
        self.w_rat = w_rationale
        self.w_tok = w_token_cost
        self.w_time = w_time_cost
        self.w_rouge_ans = w_rouge_ans
        self.model = sbert_model

    def _calculate_keyword_accuracy(self, prediction, reference):
        """Keyword matching for brief response."""
        pred_words = set(prediction.strip().lower().split())
        ref_words = set(reference.strip().lower().split())
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'of', 'and', 'is', 'are', 'by'}
        important_ref_words = ref_words - stop_words

        if not important_ref_words:
            return 1.0 if reference.strip().lower() in prediction.strip().lower() else 0.0

        matches = len(important_ref_words.intersection(pred_words))
        coverage = matches / len(important_ref_words)
        return 1.0 if coverage == 1.0 else 0.0

    def _calculate_semantic_loss(self, gen_text, ref_text):
        """
        Calculate Loss based on SBERT Cosine Similarity.
        Similarity ranges from -1 (opposite) to 1 (identical).
        Loss = 1 - Similarity.
        Loss Range: 0 (perfect) to 2 (completely different).
        """
        if not gen_text.strip():
            return 1.0 # Default penalty for empty text

        # Encoding
        embeddings1 = self.model.encode(gen_text, convert_to_tensor=True)
        embeddings2 = self.model.encode(ref_text, convert_to_tensor=True)

        # Cosine Similarity
        cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()

        # Transform into Loss (Minimize)
        # If score is 1.0 (identical) -> Loss 0.0
        # If score is 0.0 (completely different) -> Loss 1.0
        return 1.0 - cosine_score

    def compute(self, gen_ans, target_ans, gen_rat, target_rat, generation_time):

        # --- 1. Brief answer ---
        accuracy = self._calculate_keyword_accuracy(gen_ans, target_ans)
        rouge_ans = rouge_metric.compute(predictions=[gen_ans], references=[target_ans])['rougeL']

        # Answer Costs
        error_cost = (1.0 - accuracy) * self.w_acc * 5.0 # Harsh error penalty
        ans_div_cost = (1.0 - rouge_ans) * self.w_rouge_ans
        token_cost = len(gen_ans.split()) * self.w_tok

        # --- 2. REASONING (Rationale) - Powered by SBERT ---
        # Here we use SBERT instead of ROUGE
        semantic_divergence = self._calculate_semantic_loss(gen_rat, target_rat)
        rat_cost = semantic_divergence * self.w_rat

        # --- 3. TOTAL ---
        total_loss = error_cost + ans_div_cost + rat_cost + token_cost + (self.w_time * generation_time)

        return {
            "loss": round(total_loss, 4),
            "details": {
                "is_correct": accuracy > 0,
                "error_cost": round(error_cost, 4),
                "rat_semantic_loss": round(rat_cost, 4), # Costo semantico SBERT
                "raw_similarity": round(1.0 - semantic_divergence, 4), # Quanto si somigliano (0-1)
                "ans_struct_loss": round(ans_div_cost, 4),
                "token_loss": round(token_cost, 4)
            }
        }


if __name__ == "__main__":
    # 0. load ROUGE
    rouge_metric = evaluate.load("rouge")
    
    # 1. load SBERT
    print("Caricamento modello SBERT (potrebbe richiedere qualche secondo)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT caricato.")
    
    # 2. Initialize Cost Calculator with SBERT
    engine_sbert = CostCalculatorSBERT(
        w_accuracy=2.0,       
        w_rationale=2.0,      
        w_token_cost=0.05,    
        w_time_cost=0.5,      
        w_rouge_ans=1.0      
    )
    print("Engine SBERT pronto.")
    
    # Example usage
    generated_answer = "Plants"
    target_answer = "from plants"
    bad_rationale = "Plants are recreational drugs and medicine states that author found caffeine in the context provided."
    good_rationale = "Based on the text, drugs like morphine and caffeine originate directly from plants. Therefore, the correct answer is that they come from plants."
    target_rationale = "The article states that many medicinal and recreational drugs, such as tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come directly from plants. These are some examples of the medicines found in plants mentioned by the author. Thus it can be stated with certainty that some medicines do indeed come from plants. Therefore, \"from plants\" is the correct answer option to this question based on the context provided."
    time_taken = 1.5  # seconds

    # 3. Compute fitness
    result = engine_sbert.compute(generated_answer, target_answer, good_rationale, target_rationale, time_taken)
    print("Good - rationale | Loss Result:", result)
    
    result = engine_sbert.compute(generated_answer, target_answer, bad_rationale, target_rationale, time_taken)
    print("Bad - rationale | Loss Result:", result)