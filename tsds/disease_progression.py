import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import skipgrams


def merge_and_sort_all_visits(
    df_clustering, df_prediction, ids_col: str, timestamp_col: str
):
    return (
        pd.concat([df_clustering, df_prediction])
        .sort_values(by=[ids_col, timestamp_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def group_all_clusters_by_patient(df, ids_col: str):
    """Returns df with groupped patient's visit and counted number of visits/clusters per patient

    Args:
        df (pandas df): All patients' visits and corresponding clusters
        ids_col (str): Name of the column we are groupping clusters column by

    Returns:
        pandas df: df containing id and list of groupped clusters, as well as count of total visits
    """
    df_groupped = pd.DataFrame(
        df.groupby(ids_col)["cluster"].apply(list).reset_index(name="clusters")
    )
    df_groupped["no_visits"] = df_groupped.apply(
        lambda row: len(row["clusters"]), axis=1
    )
    return df_groupped


def get_subset_by_number_of_visits(df, ids_col, visits_from, visits_to):
    return df.loc[(df["no_visits"] >= visits_from) & (df["no_visits"] <= visits_to)]


def plot_number_of_visits_distribution(df, visits_from, visits_to):
    # Informational plot about visit frequency
    visits_lengths = sorted(list(df["no_visits"]))
    occurrence = {item: visits_lengths.count(item) for item in visits_lengths}

    bars = plt.bar(occurrence.keys(), occurrence.values())

    for i in range(visits_from, visits_to):
        bars.get_children()[i].set_color("green")

    plt.xlabel("Number of total visits")
    plt.ylabel("Number of patients")
    plt.title("Number of visits at the doctor per patient")
    plt.show()


def get_skipgrams_for_k_n(df, n: int, k: int):
    """Counts skipgrams for given skipgram lengths (n) and skip numbers (k)"""
    skipgram_counts = {}

    # Iter over df with patient visit PD disease clusters
    for index, row in df.iterrows():  # TODO refactor iterrows
        visits_arr = row["clusters"]
        for skipgram in skipgrams(visits_arr, n=n, k=k):
            skipgram_counts[skipgram] = (
                skipgram_counts.get(skipgram, 0) + 1
            )  # increase count of skipgram for 1

    skipgram_counts_sorted = dict(
        sorted(skipgram_counts.items(), key=lambda x: x[1], reverse=True)
    )
    return skipgram_counts_sorted


def map_tuples_to_strings(tuples):
    """Returns tuples of numbers as concatenated strings"""
    return ["".join(map(str, tpl)) for tpl in tuples]


def skip_gram_progress_analysis_charts(df, max_n):
    # This function should returns and plots all ranges k-skip-n-grams
    skipgrams_counts_all = {}  # counted skipgrams for various n/k values
    for n in range(2, max_n):
        for k in range(n + 1):
            skipgrams_counts_all[f"k_{k}_n_{n}"] = get_skipgrams_for_k_n(df, n, k)

    # Plot skipgram group change frequencies
    nrows, ncols = max_n, max_n - 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    ax_index = 0  # tracks which axis we are on

    for n in range(2, max_n):  # n-gram lengths
        for k in range(n + 1):  # skips
            skipgram_count_for_kn = skipgrams_counts_all[f"k_{k}_n_{n}"]
            axes[ax_index // ncols, ax_index % ncols].bar(
                map_tuples_to_strings(list(skipgram_count_for_kn.keys())[:10]),
                list(skipgram_count_for_kn.values())[:10],
                width=0.8,
            )
            axes[ax_index // ncols, ax_index % ncols].title.set_text(
                f"{k}-skip-{n}-grams"
            )
            axes[ax_index // ncols, ax_index % ncols].tick_params(
                axis="x", labelrotation=90
            )
            ax_index += 1

    return skipgrams_counts_all


def markov_chain_progress(bigrams, cluster_no):
    markow_chain_mtx = np.zeros((cluster_no, cluster_no))

    for i in range(cluster_no):
        for j in range(cluster_no):
            markow_chain_mtx[i, j] = bigrams.get((i, j), 0)
    m_chain_df = pd.DataFrame(markow_chain_mtx)
    return m_chain_df.div(m_chain_df.sum(axis=1), axis=0).round(4)
