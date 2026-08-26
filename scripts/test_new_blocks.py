"""Exercise the two added blocks against hand-computable fixtures.

Neither block is reachable without the full model table, so the logic is
checked here in isolation: the deviance comparison against scikit-learn's own
formula, and the covariance term against a case where the answer is known.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

TARGET_VOLUME = "n_pickups"
TARGET_VALUE = "mean_total"

# --- Deviance is defined on every row a log-link GLM can produce -----------
# Zero counts are the case that breaks a naive y*log(y/mu) implementation.
observed = np.array([0.0, 0.0, 5.0, 300.0])
fitted = np.array([0.4, 12.0, 4.0, 280.0])
value = mean_poisson_deviance(observed, fitted)
assert np.isfinite(value), "deviance must be finite when y contains zeros"
# `np.where` evaluates both branches, so the zero counts still reach
# `log(0/mu)` and raise a divide-by-zero warning even though the result is
# discarded. Compute the log term only where it is defined.
log_term = np.zeros_like(observed)
positive = observed > 0
log_term[positive] = observed[positive] * np.log(
    observed[positive] / fitted[positive]
)
manual = 2 * np.mean(log_term - (observed - fitted))
assert abs(value - manual) < 1e-9, (value, manual)
print(f"deviance handles zero counts: {value:.4f} == {manual:.4f}")

# A model that predicts the mean everywhere must lose to one that does not.
flat = np.full_like(observed, observed.mean())
assert mean_poisson_deviance(observed, flat) > value
print("deviance ranks a fitted model above an intercept-only one")

# --- The covariance term, on a case with a known answer -------------------
# Construct residuals that co-vary exactly. If the count residual is c and the
# fare residual is f, the mean product is the empirical covariance about zero.
predictions = pd.DataFrame({
    TARGET_VOLUME: [10.0, 20.0, 30.0, 0.0],
    "pred_pickups": [8.0, 18.0, 26.0, 0.5],
    TARGET_VALUE: [50.0, 60.0, 70.0, np.nan],
    "pred_mean_total": [49.0, 58.0, 66.0, 55.0],
    "sum_total_amount": [500.0, 1200.0, 2100.0, 0.0],
})

defined = predictions[TARGET_VALUE].notna()
count_residual = (
    predictions.loc[defined, TARGET_VOLUME]
    - predictions.loc[defined, "pred_pickups"]
).to_numpy(dtype=float)
fare_residual = (
    predictions.loc[defined, TARGET_VALUE]
    - predictions.loc[defined, "pred_mean_total"]
).to_numpy(dtype=float)
residual_covariance = float(np.mean(count_residual * fare_residual))

# Residuals are (2, 2, 4) and (1, 2, 4); mean product = (2 + 4 + 16) / 3.
expected = (2 * 1 + 2 * 2 + 4 * 4) / 3
assert abs(residual_covariance - expected) < 1e-12, (
    residual_covariance, expected
)
assert not np.isnan(residual_covariance), "the empty hour must not poison it"
print(f"covariance term: {residual_covariance:.4f} == {expected:.4f}, "
      "empty hour excluded")

# The sign convention has to match the shortfall's: both positive means the
# combination under-predicts, so the two are comparable in the printed block.
observed_total = float(predictions["sum_total_amount"].sum())
predicted_total = float(
    (predictions["pred_pickups"] * predictions["pred_mean_total"]).sum()
)
shortfall_per_hour = (observed_total - predicted_total) / len(predictions)
assert shortfall_per_hour > 0 and residual_covariance > 0, (
    "under-prediction must give both quantities the same sign"
)
print(f"signs agree: covariance {residual_covariance:+.1f}, "
      f"shortfall {shortfall_per_hour:+.1f} per hour")

# --- The winner-selection branch ------------------------------------------
volume_metrics = {
    "poisson": {"rmse": 40.644},
    "negative_binomial": {"rmse": 44.086},
}
deviance = {"poisson": 9.0, "negative_binomial": 7.0}
MODEL_NAMES = {"poisson": "Poisson", "negative_binomial": "negative binomial"}
deviance_winner = min(deviance, key=deviance.get)
rmse_winner = min(
    ("poisson", "negative_binomial"),
    key=lambda key: volume_metrics[key]["rmse"],
)
assert rmse_winner == "poisson" and deviance_winner == "negative_binomial"
assert deviance_winner != rmse_winner, "this fixture must take the split branch"
print(f"split branch reached: RMSE -> {MODEL_NAMES[rmse_winner]}, "
      f"deviance -> {MODEL_NAMES[deviance_winner]}")

# And the agreeing case must take the other branch.
deviance = {"poisson": 7.0, "negative_binomial": 9.0}
assert min(deviance, key=deviance.get) == rmse_winner
print("agreeing branch reached when both losses point the same way")

print("\nAll fixtures passed.")
