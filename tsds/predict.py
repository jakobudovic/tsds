import numpy as np
import shap
import sklearn  # del?
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
        df_bl_visits_clusters[[f"no_cluster_{optimal_cluster_no}"]].copy(),
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, np.ravel(y), test_size=0.3, random_state=22
    )

    model, explainer = get_best_performing_model(
        X_train, X_test, y_train, y_test, nn_max_iter
    )
    print(f"Best performing model for prediction: {model}")

    y_pred = model.predict(df_prediction.drop([ids_col], axis=1).values)
    df_prediction.loc[:, "cluster"] = np.ravel(y_pred)  # append predictions
    df_clustering.loc[:, "cluster"] = np.ravel(y)  # append predictions

    return df_clustering, df_prediction, X_train, X_test, model, explainer


def get_best_performing_model(
    X_train,
    X_test,
    y_train,
    y_test,
    nn_max_iter=150,
):

    # 1. Neural network
    nn_clf = MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(6, 2),
        random_state=1,
        max_iter=nn_max_iter,
    )
    nn_clf.fit(
        X_train.values, y_train
    )  # Use X_train.values for SHAP compatibility (avoids warning)
    y_test_pred = nn_clf.predict(X_test.values)
    accuracy_nn = 100 * accuracy_score(y_test, y_test_pred)

    print(f"Accuracy on test set for Neural network: {accuracy_nn:.3f}%")

    # 2. SVC
    svc_clf = sklearn.svm.SVC(kernel="linear", probability=True)
    svc_clf.fit(X_train.values, y_train)
    y_test_pred = svc_clf.predict(X_test.values)

    accuracy_svc = 100 * accuracy_score(y_test, y_test_pred)
    print(f"Accuracy on test set for SVC: {accuracy_svc:.3f}%")

    # 3. Random forest
    rf_clf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf_clf.fit(X_train.values, y_train)
    y_test_pred = rf_clf.predict(X_test.values)

    accuracy_rf = 100 * accuracy_score(y_test, y_test_pred)
    print(
        f"Accuracy on test set for Random forest: {100 * accuracy_score(y_test, y_test_pred):.3f}%"
    )

    models_with_acc = {
        accuracy_nn: (nn_clf, shap.KernelExplainer),
        accuracy_svc: (svc_clf, shap.KernelExplainer),
        accuracy_rf: (rf_clf, shap.KernelExplainer),
    }

    # Return model with the highest accuracy
    return models_with_acc[max(models_with_acc.keys())]
