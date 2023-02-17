import numpy as np
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

# shap.initjs()


def get_predicted_df(
    df_clustering,
    df_bl_visits_clusters,
    df_prediction,
    optimal_cluster_no,
    ids_col,
    nn_max_iter,
):

    X, y = (
        df_clustering.drop([ids_col], axis=1).copy(),
        np.ravel(df_bl_visits_clusters[[f"no_cluster_{optimal_cluster_no}"]].copy()),
    )

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=22)

    model, explainer = get_best_performing_model(X.values, y, nn_max_iter)
    print(f"Best performing model for prediction: {model}")

    model.fit(X.values, y)
    y_pred = model.predict(df_prediction.drop([ids_col], axis=1).values)
    df_prediction.loc[:, "cluster"] = np.ravel(y_pred)  # append predictions
    df_clustering.loc[:, "cluster"] = np.ravel(y)  # append predictions

    return df_clustering, df_prediction, X_train, X_test, model, explainer


def get_best_performing_model(X, y, nn_max_iter=150):

    # 1. Neural network
    nn_clf = MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(6, 2),
        random_state=1,
        max_iter=nn_max_iter,
    )
    accuracy_nn = np.mean(cross_val_score(nn_clf, X, y, cv=10))
    print(f"Accuracy on test set for Neural network: {accuracy_nn * 100:.2f}%")

    # 2. SVC
    svc_clf = sklearn.svm.SVC(kernel="linear", probability=True)
    accuracy_svc = np.mean(cross_val_score(svc_clf, X, y, cv=10))
    print(f"Accuracy on test set for SVC: {accuracy_svc * 100:.2f}%")

    # 3. Random forest
    rf_clf = RandomForestClassifier(n_estimators=20, random_state=42)
    accuracy_rf = np.mean(cross_val_score(rf_clf, X, y, cv=10))
    print(f"Accuracy on test set for Random forest: {accuracy_rf * 100:.2f}%")

    models_with_acc = {
        accuracy_nn: (nn_clf, shap.KernelExplainer),
        accuracy_svc: (svc_clf, shap.KernelExplainer),
        accuracy_rf: (rf_clf, shap.KernelExplainer),
    }

    # Return model with the highest accuracy
    return models_with_acc[max(models_with_acc.keys())]
