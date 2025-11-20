import evaluate
import numpy as np


class CostCalculator:
    def __init__(self,
                 w_accuracy=10.0,     # HUGE penalty in case of error
                 w_token_cost=0.05,   # Cost per word used
                 w_time_cost=0.5,     # Cost per second elapsed
                 w_divergence=2.0):   # Cost related to different structure w.r.t. reference solution

        self.w_err = w_accuracy
        self.w_tok = w_token_cost
        self.w_time = w_time_cost
        self.w_div = w_divergence

    def calculate_keyword_accuracy(self, prediction, reference):
        """
        Returns 1 if correct, 0 otherwise
        """
        pred_words = set(prediction.strip().lower().split())
        ref_words = set(reference.strip().lower().split())
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'of', 'and', 'is', 'are'}
        important_ref_words = ref_words - stop_words

        if not important_ref_words:
            return 1.0 if reference.strip().lower() in prediction.strip().lower() else 0.0

        matches = len(important_ref_words.intersection(pred_words))
        coverage = matches / len(important_ref_words)
        return 1.0 if coverage == 1.0 else 0.0

    def compute(self, generated_text, target_text, generation_time):
        """
        Calculate the totale LOSS (to Minimize)
        Theoretical objective: 0.0 (actually impossible, but we tend to it)
        """
        # 1. Accuracy
        accuracy = self.calculate_keyword_accuracy(generated_text, target_text)
        # Transform into error (0 if correct, 1 if wrong)
        error_rate = 1.0 - accuracy

        # 2. ROUGE (Structure)
        rouge_results = rouge_metric.compute(predictions=[generated_text], references=[target_text])
        rouge_l = rouge_results['rougeL']

        # GATING: If answer is wrong (Error=1). We do not care of the structure.
        # We consider divergence = 1, othewise we compute it (1-ROUGE)
        # Note: if accuracy is 0, the divergence is not helpful, the error dominates 
        structural_divergence = 1.0 - rouge_l

        # 3. Cost of resources (time consumed)
        token_count = len(generated_text.split())

        # --- LOSS COMPUTATION ---

        # Base Cost = (Error Weight * Error) + (Divergence Weight * Divergence)
        # If accuracy = 1 (correct), the first term vanishes.
        base_cost = (self.w_err * error_rate) + (self.w_div * structural_divergence)

        # Efficiency cost = Token + time
        efficiency_cost = (self.w_tok * token_count) + (self.w_time * generation_time)

        total_loss = base_cost + efficiency_cost

        return {
            "loss": round(total_loss, 4),
            "details": {
                "is_correct": accuracy > 0,
                "error_cost": round(self.w_err * error_rate, 4),
                "divergence_cost": round(self.w_div * structural_divergence, 4),
                "token_cost": round(self.w_tok * token_count, 4),
                "time_cost": round(self.w_time * generation_time, 4)
            }
        }


# Example usage
if __name__ == "__main__":
    rouge_metric = evaluate.load('rouge')

    generated = "Paris is the capital city of France."
    target = "Paris is the capital city of France."
    time_taken = 1.2  # secondi

    # Init
    loss_engine = CostCalculator(
        w_accuracy=10.0,       # Main goal: NO ERRORS
        w_token_cost=0.05,  # Prefer brief answers
        w_time_cost=0.2,    # Prefer quick answers
        w_divergence=1.0    # Encourage similar prompt structures (proposed vs. groundtruth) 
    )

    result = loss_engine.compute(generated, target, time_taken)
    print("Loss Result:", result)