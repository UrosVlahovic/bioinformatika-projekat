import pandas as pd
import numpy as np
from cyvcf2 import VCF

def load_genotypes(path, max_variants=None):
    vcf = VCF(path)

    samples = vcf.samples
    rows = []
    index = []

    for i, variant in enumerate(vcf):
        if max_variants is not None and i >= max_variants:
            break


        if len(variant.ALT) != 1:
            continue

        genotypes = []

        for genotype in variant.genotypes:
            a1 = genotype[0]
            a2 = genotype[1]

            # ako nešto fali, tretiram kao 0
            if a1 == -1 or a2 == -1:
                alt_count = 0
            else:
                alt_count = a1 + a2

            genotypes.append(alt_count)

        rows.append(genotypes)
        index.append(f"{variant.CHROM}-{variant.POS}")

    df = pd.DataFrame(rows, columns=samples, index=index, dtype=np.uint8)
    return df

from scipy.stats import chi2_contingency


def chi_square_model(geno_row, y, model):
    if model == "dominant":
        groups = (geno_row > 0).astype(int)
        table = np.zeros((2, 2), dtype=float)

    elif model == "additive":
        groups = geno_row.astype(int)
        table = np.zeros((3, 2), dtype=float)

    else:
        raise ValueError("Nepoznat model")

    for g, d in zip(groups, y):
        table[g, d] += 1

    table += 1  # Laplace korekcija

    _, p, _, _ = chi2_contingency(table)
    return p


def gwas_analysis(geno_df, y, bonferroni=True):
    """
    Izvršava GWAS analizu nad genotipskim podacima.

    Implementirana su dva GWAS modela:

    1. Dominantni model:
       Genotip 0 se poredi protiv genotipova 1 i 2.
       Drugim rečima, razlikuju se osobe koje nemaju alternativni alel
       od osoba koje imaju bar jedan alternativni alel.
       Pogodan je jer detektuje varijante kod kojih je
       prisustvo makar jednog alternativnog alela dovoljno da utiče na fenotip,
       što je čest slučaj kod retkih varijanti.

    2. Aditivni model:
       Koriste se sve tri genotipske kategorije: 0, 1 i 2.
       Ovaj model je izabran jer čuva više informacija od dominantnog modela
       i omogućava da se vidi da li broj alternativnih alela utiče na bolest.
    """

    dominant_p_values = []
    additive_p_values = []

    for i in range(len(geno_df)):
        row = geno_df.iloc[i].values

        dominant_p_values.append(chi_square_model(row, y, "dominant"))
        additive_p_values.append(chi_square_model(row, y, "additive"))

    result = pd.DataFrame({
        "dominant_p": dominant_p_values,
        "additive_p": additive_p_values
    }, index=geno_df.index)

    result = result.sort_values("dominant_p")

    if bonferroni:
        n = len(geno_df)
        threshold = 0.05 / n

        result = result[
            (result["dominant_p"] < threshold) |
            (result["additive_p"] < threshold)
        ]

    return result


import matplotlib.pyplot as plt

def manhattan_plot(gwas_df):
    # uzimam jednu kolonu (npr dominantni model)
    p_values = gwas_df["dominant_p"]

    # -log10(p)
    log_p = -np.log10(p_values)

    # x osa = indeks (pozicije)
    x = range(len(log_p))

    plt.figure(figsize=(10,5))
    plt.scatter(x, log_p, s=10)

    # Bonferroni linija
    n = len(gwas_df)
    threshold = -np.log10(0.05 / n)
    plt.axhline(y=threshold, color='r', linestyle='--')

    plt.xlabel("Genomic position")
    plt.ylabel("-log10(p)")
    plt.title("Manhattan plot")

    plt.show()

def discretize_pvalues(p_values):
    """
    Pretvara p-vrednosti u diskretne kategorije na osnovu -log10(p).
    Veća vrednost znači jači GWAS signal.
    """
    log_p = -np.log10(p_values)

    obs = []
    for x in log_p:
        if x < 2:
            obs.append(0)   # slab signal
        elif x < 4:
            obs.append(1)   # srednji signal
        elif x < 6:
            obs.append(2)   # jak signal
        else:
            obs.append(3)   # vrlo jak signal

    return np.array(obs)


def viterbi(obs):


    # matrica prelaza između stanja
    A = np.array([
        [0.99, 0.01],   # background uglavnom ostaje background
        [0.10, 0.90]    # signal uglavnom ostaje signal
    ])

    # matrica emisija: koliko je verovatno da stanje emituje određenu kategoriju
    E = np.array([
        [0.80, 0.15, 0.04, 0.01],  # background češće daje slab signal
        [0.05, 0.15, 0.30, 0.50]   # signal češće daje jak signal
    ])

    # početne verovatnoće stanja
    pi = np.array([0.99, 0.01])

    n = len(obs)

    dp = np.zeros((n, 2))
    prev = np.zeros((n, 2), dtype=int)

    dp[0] = np.log(pi) + np.log(E[:, obs[0]])

    for i in range(1, n):
        for state in range(2):
            values = dp[i - 1] + np.log(A[:, state])
            prev[i, state] = np.argmax(values)
            dp[i, state] = np.max(values) + np.log(E[state, obs[i]])

    states = np.zeros(n, dtype=int)
    states[-1] = np.argmax(dp[-1])

    for i in range(n - 2, -1, -1):
        states[i] = prev[i + 1, states[i + 1]]

    return states


def hmm_analysis(gwas_result):

    p_values = gwas_result["dominant_p"].values

    obs = discretize_pvalues(p_values)
    states = viterbi(obs)

    hmm_result = pd.DataFrame({
        "position": gwas_result.index,
        "dominant_p": p_values,
        "observation": obs,
        "state": states
    })

    return hmm_result


def extract_regions(hmm_result):

    regions = []
    start = None
    start_idx = None

    for i in range(len(hmm_result)):
        state = hmm_result.iloc[i]["state"]

        if state == 1 and start is None:
            start = hmm_result.iloc[i]["position"]
            start_idx = i

        elif state == 0 and start is not None:
            end = hmm_result.iloc[i - 1]["position"]
            end_idx = i - 1

            region_data = hmm_result.iloc[start_idx:end_idx + 1]
            best_row = region_data.loc[region_data["dominant_p"].idxmin()]

            regions.append({
                "start": start,
                "end": end,
                "n_snps": len(region_data),
                "best_position": best_row["position"],
                "best_p": best_row["dominant_p"]
            })

            start = None
            start_idx = None

    if start is not None:
        end = hmm_result.iloc[-1]["position"]
        end_idx = len(hmm_result) - 1

        region_data = hmm_result.iloc[start_idx:end_idx + 1]
        best_row = region_data.loc[region_data["dominant_p"].idxmin()]

        regions.append({
            "start": start,
            "end": end,
            "n_snps": len(region_data),
            "best_position": best_row["position"],
            "best_p": best_row["dominant_p"]
        })

    return pd.DataFrame(regions)

if __name__ == "__main__":
    geno_df = load_genotypes("1KG.subset.bcf", max_variants=100)

    pheno = pd.read_csv("status/status_1.tsv", sep="\t")
    y = pheno["disease"].values

    gwas_result = gwas_analysis(geno_df, y, bonferroni=False)
    print("GWAS rezultati:")
    print(gwas_result.head())

    gwas_significant = gwas_analysis(geno_df, y, bonferroni=True)
    print("Bonferroni značajne pozicije:")
    print(gwas_significant)

    manhattan_plot(gwas_result)

    hmm_result = hmm_analysis(gwas_result)
    regions_df = extract_regions(hmm_result)

    print("HMM rezultati:")
    print(hmm_result.head())

    print("Broj stanja:")
    print(hmm_result["state"].value_counts())

    print("Detektovani regioni:")
    print(regions_df)