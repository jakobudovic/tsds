import matplotlib.pyplot as plt
import shap


def explain_model_shap(
    model,
    explainer,
    X_train,
    X_test,
    explain_n=None,
    xtrain_samples=50,
    xtest_samples=150,
):
    if explain_n and explain_n > X_test.shape[0]:
        print(
            f"Tried to explain too many examples... Setting {explain_n} to {X_test.shape[0]}"
        )
        explain_n = X_test.shape[0]

    _exiplainer = explainer(
        model.predict_proba,
        shap.sample(X_train, xtrain_samples),  # Represent data
    )
    shap_values = _exiplainer.shap_values(
        X_test.sample(explain_n or X_test.shape[0], random_state=42),
        nsamples=xtest_samples,
    )

    # Plot all shap values importance
    shap.summary_plot(
        shap_values,
        X_test.sample(explain_n or X_test.shape[0], random_state=42),
        plot_type="bar",
        plot_size=(8, 5),
    )

    # Plot shap values
    for i, values in enumerate(shap_values):
        shap.summary_plot(
            values,
            X_test.sample(explain_n or X_test.shape[0], random_state=42),
            plot_type="violin",
            plot_size=(8, 5),
            show=False,
        )
        plt.title(f"Class = {i}")
        plt.show()
