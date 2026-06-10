from scipy.stats import binom

k = 18
n = 122
alpha = 0.05
p = 1 / k
c = binom.isf(alpha, n, p) + 1  # Inverse survival function + 1
threshold = (c / n) * 100

print(f"Threshold: {threshold:.2f}% ({c}/{n})")

# Level 02 Hops: Accuracy 6.12% (312/5101)
# Level 03 Hops: Accuracy 6.11% (312/5103)
# Level 04 Hops: Accuracy 6.12% (312/5101)
# Level 05 Hops: Accuracy 9.19% (17/185)
# Level 06 Hops: Accuracy 10.48% (11/105)
# Level 07 Hops: Accuracy 9.68% (15/155)
# Level 08 Hops: Accuracy 9.63% (13/135)
# Level 09 Hops: Accuracy 9.68% (12/124)
# Level 10 Hops: Accuracy 9.84% (12/122)