"""Step 05: fit the two-class latent class model and assign rapid/slow progression labels."""

import warnings

from lca_utils import (
    DEFAULT_INIT_SEEDS,
    build_discrete_time_matrix,
    build_posterior_table,
    class_expected_event_codes,
    fit_stable_model,
    load_lca_input,
)


INPUT_FILE = "PPMI_2_time_to_event.csv"
OUTPUT_FILE = "PPMI_3_patients_subtypes_lca_k2.csv"
N_COMPONENTS = 2
MAX_ITER = 1000

warnings.filterwarnings("ignore")


def main() -> None:
    """Fit the final model and save each patient's class and posterior probabilities."""
    frame, time_cols = load_lca_input(INPUT_FILE)
    matrix = build_discrete_time_matrix(frame, time_cols)
    print(
        f">>> N={len(matrix)}, milestones={len(time_cols)}, "
        f"event-history indicators={matrix.shape[1]}"
    )
    print(
        f">>> Fitting censoring-aware k={N_COMPONENTS} model with "
        f"{len(DEFAULT_INIT_SEEDS)} independent starts..."
    )
    stable_fit = fit_stable_model(matrix, N_COMPONENTS, max_iter=MAX_ITER)
    model = stable_fit.model
    output = build_posterior_table(frame, matrix, model, time_cols)

    expectations = class_expected_event_codes(model, time_cols)
    subtype_by_class = output.groupby("Subtype_Label")["Subtype"].first()
    expectations["Subtype"] = expectations["Class"].map(subtype_by_class)
    print(
        "\n--- Model-implied event-time codes (lower indicates faster progression) ---"
    )
    print(
        expectations.to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    counts = output["Subtype"].value_counts().reindex(["Slow", "Rapid"])
    print("\n--- Final class assignment ---")
    for subtype, count in counts.items():
        print(f"{subtype}: {int(count)} ({count / len(output) * 100:.1f}%)")
    print(
        f">>> BIC={model.bic(matrix):.2f}; best seed={stable_fit.best_seed}; "
        f"converged starts={stable_fit.converged_starts}/{len(DEFAULT_INIT_SEEDS)}; "
        f"minimum stability ARI={stable_fit.min_stability_ari:.4f}"
    )

    output.to_csv(OUTPUT_FILE, encoding="utf-8-sig")
    print(f">>> Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
