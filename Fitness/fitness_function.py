import numpy as np
from Utils.LLM import LLM

class UnifiedFitnessCalculator:
    """
    Calculates evolutionary score dynamically.
    Focuses on Answer Accuracy and a Self-Adjusting Token Penalty.
    """
    def __init__(self,
                 llm: LLM,
                 accuracy_score=1.0,         
                 max_penalty=0.9): 
        self.acc_score = accuracy_score
        self.max_penalty = max_penalty
        self.llm = llm
        
        # Initial safe baselines for Generation 0
        self.target_mean = 200.0
        self.target_std = 25.0   

    def update_baselines(self, token_usages: list[int]):
        """
        Calculates the dynamic token threshold based on the population's 
        usage, utilizing IQR to strip out hallucination outliers.
        """
        if not token_usages:
            return
        
        arr = np.array(token_usages)
        
        # 1. Filter outliers using Interquartile Range (IQR)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        lower_bound_filter = q1 - 1.5 * iqr
        upper_bound_filter = q3 + 1.5 * iqr
        
        # Keep only "normal" token usages
        filtered = arr[(arr >= lower_bound_filter) & (arr <= upper_bound_filter)]
        
        # 2. Update Mean and Std Dev (fallback to raw array if filtering stripped everything)
        if len(filtered) > 0:
            self.target_mean = np.mean(filtered)
            self.target_std = max(1.0, np.std(filtered)) # Force > 0 to prevent division by zero
        else:
            self.target_mean = np.mean(arr)
            self.target_std = max(1.0, np.std(arr))
            
        print(f"   📊 Dynamic Token Baseline Shifted -> Mean: {self.target_mean:.1f} | Std: {self.target_std:.1f}")

    def compute_score(self, is_correct: bool, token_count: int) -> float:
        """
        Calculates the raw evolutionary score for a SINGLE problem.
        Higher score is better.
        """
        # 1. Answer Score 
        ans_points = self.acc_score if is_correct else 0.0

        # 2. Dynamic Token Penalty (Smooth Linear Interpolation)
        lower_bound = self.target_mean - self.target_std
        upper_bound = self.target_mean + self.target_std
        
        if token_count <= lower_bound:
            token_penalty = 0.0
        elif token_count >= upper_bound:
            token_penalty = self.max_penalty
        else:
            # Scale proportionally between 0.0 and max_penalty
            token_penalty = self.max_penalty * ((token_count - lower_bound) / (upper_bound - lower_bound))

        # Total final score for this problem (ensuring it never goes negative here)
        return max(0.0, (ans_points - token_penalty))