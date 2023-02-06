from tsds.clustering import get_optimal_cluster_no
from tsds.disease_progression import (
    get_subset_by_number_of_visits,
    group_all_clusters_by_patient,
    markov_chain_progress,
    merge_and_sort_all_visits,
    plot_number_of_visits_distribution,
    skip_gram_progress_analysis_charts,
)
from tsds.explain import explain_model_shap
from tsds.nmf import get_NMF
from tsds.predict import get_predicted_df

# df_prediction = X_norm[~X_norm.index.isin(X.index)].copy()

# old: TimingDataSegmentation
# new: TimeSeriesDataSegmentation
class TimeSeriesDataSegmentation:
    def __init__(self, df_clustering, df_prediction):
        print(f"TimeSeriesDataSegmentation object created!")
        self.df_clustering = df_clustering
        self.df_prediction = df_prediction

    # 1. NMF data dimension reducionality
    def get_nmf_data(self, ids_col, timestamp_col, nmf_n_components, nmf_max_iter):
        _, self.df_clustering_nmf = get_NMF(
            self.df_clustering.drop([ids_col, timestamp_col], axis=1),
            n_components=nmf_n_components,
            max_iter=nmf_max_iter,
        )
        return self.df_clustering_nmf

    # 2. Get optimal number of clusters and df with the best prediction
    def optimal_cluster_no(self, df_clustering_nmf, clusters_min, clusters_max):
        return get_optimal_cluster_no(df_clustering_nmf, clusters_min, clusters_max)

    # 3. Predicting other visits groups
    def predict_df(self, *args):  # TODO maybe type out arguments
        return get_predicted_df(*args)

    # 4. Explaining subgroups
    def explain_model_shap(
        self,
        model,
        explainer,
        X_train,
        X_test,
        explain_n,
        xtrain_samples,
        xtest_samples,
    ):
        return explain_model_shap(
            model, explainer, X_train, X_test, explain_n, xtrain_samples, xtest_samples
        )

    # 5. Disease progression analysis
    def prepare_data_for_disease_progression(
        self,
        df_clustering,
        df_prediction,
        ids_col,
        timestamp_col,
        visits_from,
        visits_to,
        plot_visits_distribution=False,
    ):
        df_all_visits = merge_and_sort_all_visits(
            df_clustering, df_prediction, ids_col, timestamp_col
        )

        df_clusters_groupped = group_all_clusters_by_patient(df_all_visits, ids_col)

        df_clusters_groupped_subset = get_subset_by_number_of_visits(
            df_clusters_groupped, ids_col, visits_from, visits_to
        )

        if plot_visits_distribution:
            try:
                plot_number_of_visits_distribution(
                    df_clusters_groupped, visits_from, visits_to
                )
            except BaseException as e:
                print(f"Could not display plot number of visits distribution: {e}")

        return df_clusters_groupped_subset

    # 5.1 Skip grams
    def skip_gram_progress(self, df, max_n):
        return skip_gram_progress_analysis_charts(df, max_n)

    # 5.2 Markov chain
    def markov_chain_progress(self, bigrams, cluster_no):
        return markov_chain_progress(bigrams, cluster_no)
