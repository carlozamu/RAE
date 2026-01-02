"""
1) Poulation initialization (simple same pormt, single node, 50 individuals)
Evaluation of a single individual as baseline
Loop until termination condition (fitness under threshold [<0.1] or max time (X hours/ k-pressed) reached):
    Loop for each species:
        3) Selection (intra-species)
        4) Crossover
        5) Mutation
        6) Fitness evaluation
    Average fitness calculation per species
    7) Replacement (keep 50 individuals top for each generation)

Variabili globali:
    number of genes (for innovation number)
    generation index
    species age


Cose da loggare:
    generation number
    best-worse fitness per generation
    average fitness per species per generation
    average number of nodes and connections per individual per generation
    species count per generation
    paragone con prompt engineering

Cose da fare:
evolution manager todo
    mutazione genome max(0.1,0.9^(generation_number+1))
    mutazione gene 1/NUMERO DI NODI DEL GENOMA
    
    selection fix 

    genotype/phenotype
    new species handling


Report
    1) Abstract + Introduction (probem + why NEAT)
    2) Dataset + benchmarking
    2) Selection + Fitnes
    3) Crossover + Mutation
    3) Speciation + Evolution Management
    4) Experiments + Results
"""
