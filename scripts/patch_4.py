"""Apply the reviewed fixes to notebook 4 and the README.

Every substitution asserts the old text is present first, so that a file which
has moved on since the review fails here rather than being silently mangled.
"""

import json
from pathlib import Path

NOTEBOOK = Path("4_modelling.ipynb")
README = Path("README.md")


def cell_source(cell) -> str:
    return "".join(cell["source"])


def set_source(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def sub(text: str, old: str, new: str, label: str) -> str:
    assert old in text, f"[{label}] anchor not found"
    assert text.count(old) == 1, f"[{label}] anchor is not unique"
    return text.replace(old, new)


notebook = json.loads(NOTEBOOK.read_text())
cells = notebook["cells"]

# --------------------------------------------------------------------------
# Cell 1 — the extra import
# --------------------------------------------------------------------------
source = cell_source(cells[1])
source = sub(
    source,
    "from sklearn.inspection import permutation_importance\n",
    "from sklearn.inspection import permutation_importance\n"
    "from sklearn.metrics import mean_poisson_deviance\n",
    "cell 1 import",
)
set_source(cells[1], source)

# --------------------------------------------------------------------------
# Cell 3 — pin the airport names Step 5 refers to by name
# --------------------------------------------------------------------------
source = cell_source(cells[3])
source = sub(
    source,
    'AIRPORTS = tuple(sorted(table["airport"].unique()))\n',
    'AIRPORTS = tuple(sorted(table["airport"].unique()))\n'
    "\n"
    "# The two explanations attached to the excluded overnight cells in Step 5\n"
    "# are airport-specific facts — LaGuardia's curfew, JFK's absent\n"
    "# international arrivals — so they are keyed there by name rather than by\n"
    "# position in this tuple. Checked here so that a change of scope fails at\n"
    "# the top of the notebook instead of mislabelling a limitation near the\n"
    "# bottom of it.\n"
    'assert set(AIRPORTS) == {"JFK", "LGA"}, (\n'
    '    f"Step 5 names JFK and LGA explicitly; this table carries {AIRPORTS}"\n'
    ")\n",
    "cell 3 airport assertion",
)
set_source(cells[3], source)

# --------------------------------------------------------------------------
# Markdown 12 — the negative binomial contrast, stated as a two-loss question
# --------------------------------------------------------------------------
source = cell_source(cells[12])
source = sub(
    source,
    "If the two agree on the test set, the report can say the demand model is "
    "robust to the\nvariance assumption, which is a stronger statement than "
    "either fit alone. They are\ncompared on likelihood as well as on test "
    "error; the negative binomial's AIC is charged\nfor the α estimated above, "
    "which statsmodels treats as known.",
    "If the two agree on the test set, the report can say the demand model is "
    "robust to the\nvariance assumption, which is a stronger statement than "
    "either fit alone. If they do not,\nwhich loss each wins is itself the "
    "result.\n\nThey are compared on likelihood in sample — the negative "
    "binomial's AIC charged for the α\nestimated above, which statsmodels "
    "treats as known — and on two losses out of sample in\nStep 5: squared "
    "error, which the report quotes, and mean Poisson deviance, which is the\n"
    "loss this family is actually fitted under. The second is not decoration. "
    "A negative\nbinomial's IRLS weight is `μ / (1 + α·μ)` against the "
    "Poisson's `μ`, so it discounts\nprecisely the busy hours where squared "
    "error is decided; settling the comparison on RMSE\nalone would use a "
    "metric that has already taken a side.",
    "md 12 two-loss framing",
)
set_source(cells[12], source)

# --------------------------------------------------------------------------
# Markdown 18 — same, in the step that runs the comparison
# --------------------------------------------------------------------------
source = cell_source(cells[18])
source = sub(
    source,
    "**Do the two count models agree?** The Poisson and the negative binomial "
    "differ in exactly\none assumption. If their test errors are close, the "
    "demand model is robust to the variance\nassumption and the quasi-Poisson "
    "standard errors can be reported with confidence.\n",
    "**Do the two count models agree?** The Poisson and the negative binomial "
    "differ in exactly\none assumption. If their test errors are close, the "
    "demand model is robust to the variance\nassumption and the quasi-Poisson "
    "standard errors can be reported with confidence. If they\nare not, the "
    "question worth answering is which loss each one wins, so squared error "
    "and\nmean Poisson deviance are both computed and the model that carries "
    "the recommendations is\nchosen against a stated criterion rather than "
    "against whichever figure was printed first.\n",
    "md 18 two-loss framing",
)
set_source(cells[18], source)

# --------------------------------------------------------------------------
# Cell 19 — mean Poisson deviance beside the squared-error comparison
# --------------------------------------------------------------------------
source = cell_source(cells[19])
source = sub(
    source,
    "count_gap = abs(\n",
    "# --- A count metric for a count model --------------------------------------\n"
    "# Squared error is the loss the report quotes and also the loss that favours\n"
    "# the Poisson by construction: a negative binomial's IRLS weight is\n"
    "# mu / (1 + alpha * mu) against the Poisson's mu, so it discounts exactly the\n"
    "# busy hours where squared error is decided. Mean Poisson deviance is the\n"
    "# loss the count family is fitted under, and it is reported beside RMSE so\n"
    "# that the choice below rests on a criterion rather than on a coincidence of\n"
    "# metric. Both GLMs use a log link, so every fitted rate is strictly\n"
    "# positive and the deviance is defined on every test row.\n"
    "deviance = {\n"
    '    "poisson": float(mean_poisson_deviance(\n'
    '        predictions[TARGET_VOLUME], predictions["pred_pickups"]\n'
    "    )),\n"
    '    "negative_binomial": float(mean_poisson_deviance(\n'
    '        predictions[TARGET_VOLUME], predictions["pred_pickups_nb"]\n'
    "    )),\n"
    "}\n"
    'MODEL_NAMES = {"poisson": "Poisson", "negative_binomial": "negative binomial"}\n'
    '\nprint("\\nMean Poisson deviance on the test split (lower is better)")\n'
    "for key, label in MODEL_NAMES.items():\n"
    '    print(f"  {label:<18} {deviance[key]:8.3f}")\n'
    "\n"
    "count_gap = abs(\n",
    "cell 19 deviance block",
)
source = sub(
    source,
    'print(f"\\nThe two count models differ by {count_gap:.2f} pickups in test '
    'RMSE, "\n      f"{100 * relative_gap:.0f}% of the Poisson\'s — '
    '{verdict}.")',
    'print(f"\\nThe two count models differ by {count_gap:.2f} pickups in test '
    'RMSE, "\n      f"{100 * relative_gap:.0f}% of the Poisson\'s — '
    '{verdict}.")\n'
    "\n"
    "# Which model to carry forward is decided here, from the two losses and the\n"
    "# in-sample gap, rather than asserted in the markdown above.\n"
    "deviance_winner = min(deviance, key=deviance.get)\n"
    "rmse_winner = min(\n"
    '    ("poisson", "negative_binomial"),\n'
    '    key=lambda key: volume_metrics[key]["rmse"],\n'
    ")\n"
    "if deviance_winner == rmse_winner:\n"
    '    print(f"Both out-of-sample losses point the same way "\n'
    '          f"({MODEL_NAMES[deviance_winner]}), so its advantage is a '
    'property of "\n'
    '          "the fit rather than of the metric, and the in-sample AIC gap is '
    'the "\n'
    '          "only disagreement left to explain.")\n'
    "else:\n"
    '    print(f"The two losses disagree: {MODEL_NAMES[rmse_winner]} wins on '
    'squared "\n'
    '          f"error, {MODEL_NAMES[deviance_winner]} on deviance, and the '
    'negative "\n'
    '          f"binomial already won in sample by "\n'
    '          f"{poisson_aic - negbin_aic:,.0f} of AIC. Read that as a '
    'division of "\n'
    '          "labour rather than a contradiction — the negative binomial '
    'describes "\n'
    '          "the *distribution* better, the Poisson predicts the *level* '
    'better. "\n'
    '          "A driver choosing between two ranks acts on the level, so the '
    'Poisson "\n'
    '          "carries the recommendations and the negative binomial is '
    'reported as "\n'
    '          "the contrast that shows what that choice costs.")\n',
    "cell 19 deviance verdict",
)
set_source(cells[19], source)

# --------------------------------------------------------------------------
# Cell 20 — name the airports instead of indexing them
# --------------------------------------------------------------------------
source = cell_source(cells[20])
source = sub(
    source,
    '        (AIRPORTS[1], "the overnight curfew — nothing is scheduled because "\n'
    '                      "nothing may land, so these pickups are passengers "\n'
    '                      "still clearing from an earlier arrival and the "\n'
    '                      "one-hour lags already carry them"),\n'
    '        (AIRPORTS[0], "genuine overnight arrivals, of which the international "\n'
    '                      "ones are absent from BTS entirely — this is the "\n'
    '                      "coverage gap, and it is a limitation to quote rather "\n'
    '                      "than a ratio to correlate"),\n',
    '        ("LGA", "the overnight curfew — nothing is scheduled because '
    'nothing "\n'
    '                "may land, so these pickups are passengers still clearing "\n'
    '                "from an earlier arrival and the one-hour lags already "\n'
    '                "carry them"),\n'
    '        ("JFK", "genuine overnight arrivals, of which the international '
    'ones "\n'
    '                "are absent from BTS entirely — this is the coverage gap, "\n'
    '                "and it is a limitation to quote rather than a ratio to "\n'
    '                "correlate"),\n',
    "cell 20 airports by name",
)
set_source(cells[20], source)

# --------------------------------------------------------------------------
# Markdown 25 — the omitted covariance term
# --------------------------------------------------------------------------
source = cell_source(cells[25])
source = sub(
    source,
    "**It is market revenue, not a wage.**",
    "**It omits a covariance term, and the term is measured.** The identity "
    "above holds row by\nrow. Expectations do not inherit it:\n\n"
    "`E[revenue | X] = E[pickups | X] × E[fare | X] + Cov(pickups, fare | X)`\n\n"
    "and a product of two separately fitted conditional means estimates only "
    "the first part.\nWhatever covariance survives conditioning on the "
    "features is a bias that no improvement to\neither model can remove, "
    "because neither model can see it. There is a second, smaller\nwrinkle "
    "pointing the same way: Model 2 is fitted only where `mean_total` is "
    "defined, so it\nestimates `E[fare | X, pickups > 0]` rather than "
    "`E[fare | X]`.\n\nThe cell below measures the term as the mean product of "
    "the two models' residuals and\nreports it against the shortfall it is "
    "meant to explain, so that the combination's error is\nattributed rather "
    "than charged wholesale to the demand model.\n\n"
    "**It is market revenue, not a wage.**",
    "md 25 covariance",
)
set_source(cells[25], source)

# --------------------------------------------------------------------------
# Cell 26 — measure the covariance
# --------------------------------------------------------------------------
source = cell_source(cells[26])
source = source.rstrip("\n") + "\n"
source += (
    "\n"
    "# --- What the product cannot represent -------------------------------------\n"
    "# 2c's identity `n_pickups * mean_total == sum_total_amount` holds row by\n"
    "# row; it does not carry over to expectations. The product above estimates\n"
    "# E[n | X] * E[mean | X], and the quantity it is scored against is\n"
    "# E[n * mean | X], which exceeds it by Cov(n, mean | X). That term is not a\n"
    "# defect of either fit — it is what multiplying two conditional means costs —\n"
    "# and it is measurable as the mean product of the two models' residuals.\n"
    "#\n"
    "# Measured over the hours where `mean_total` is defined, since the fare\n"
    "# residual is undefined elsewhere. The empty hours contribute nothing to the\n"
    "# covariance in any case: their revenue is zero on both sides.\n"
    "defined = predictions[TARGET_VALUE].notna()\n"
    "count_residual = (\n"
    "    predictions.loc[defined, TARGET_VOLUME]\n"
    '    - predictions.loc[defined, "pred_pickups"]\n'
    ").to_numpy(dtype=float)\n"
    "fare_residual = (\n"
    "    predictions.loc[defined, TARGET_VALUE]\n"
    '    - predictions.loc[defined, "pred_mean_total"]\n'
    ").to_numpy(dtype=float)\n"
    "residual_covariance = float(np.mean(count_residual * fare_residual))\n"
    "\n"
    "mean_observed_revenue = float(predictions['sum_total_amount'].mean())\n"
    "shortfall_per_hour = (observed_total - predicted_total) / len(predictions)\n"
    "\n"
    'print(f"\\nThe omitted covariance term, over the {int(defined.sum()):,} '
    'hours where "\n'
    '      f"{TARGET_VALUE} is defined:")\n'
    "print(f\"  Cov(n_pickups, mean_total | X)   ${residual_covariance:+,.0f} \"\n"
    '      "per airport-hour")\n'
    'print(f"  mean observed revenue            ${mean_observed_revenue:,.0f} '
    '"\n'
    '      "per airport-hour")\n'
    'print(f"  shortfall the combination leaves ${shortfall_per_hour:+,.0f} '
    '"\n'
    '      "per airport-hour")\n'
    "\n"
    "# Reported as a share of the level, which is interpretable whatever sign the\n"
    "# shortfall takes, rather than as a share of a denominator that may be near\n"
    "# zero.\n"
    "covariance_share_of_level = (\n"
    "    100 * residual_covariance / mean_observed_revenue\n"
    ")\n"
    "if abs(covariance_share_of_level) < 2.0:\n"
    '    print(f"That is {covariance_share_of_level:+.1f}% of mean revenue per '
    'hour. "\n'
    '          "The two models\' errors are close to uncorrelated once the '
    'features "\n'
    '          "are conditioned on, so the product is a sound estimator here '
    'and the "\n'
    '          "shortfall above is Model 1\'s bias rather than a structural '
    'defect of "\n'
    '          "the combination.")\n'
    "else:\n"
    '    print(f"That is {covariance_share_of_level:+.1f}% of mean revenue per '
    'hour, and "\n'
    '          f"{100 * residual_covariance / shortfall_per_hour:.0f}% of the '
    'shortfall. "\n'
    '          "Busy hours and expensive hours co-move even after the features '
    'are "\n'
    '          "accounted for, so part of the combination\'s error belongs to '
    'the "\n'
    '          "act of multiplying two conditional means and not to either fit. '
    '"\n'
    '          "Report it in the modelling section as a property of the '
    "combination, \"\n"
    '          "not in the limitations as a failure of a model.")\n'
)
set_source(cells[26], source)

# --------------------------------------------------------------------------
# Cell 32 — persist the two new quantities
# --------------------------------------------------------------------------
source = cell_source(cells[32])
source = sub(
    source,
    '        "negative_binomial_alpha": round(alpha, 4),\n'
    '        "test": volume_metrics,\n',
    '        "negative_binomial_alpha": round(alpha, 4),\n'
    '        "test": volume_metrics,\n'
    '        "test_mean_poisson_deviance": {\n'
    "            key: round(value, 4) for key, value in deviance.items()\n"
    "        },\n"
    '        "carried_into_recommendations": rmse_winner,\n',
    "cell 32 deviance json",
)
source = sub(
    source,
    '        "observed_test_revenue": round(observed_total, 2),\n'
    '        "predicted_test_revenue": round(predicted_total, 2),\n',
    '        "observed_test_revenue": round(observed_total, 2),\n'
    '        "predicted_test_revenue": round(predicted_total, 2),\n'
    '        "residual_covariance_per_hour": round(residual_covariance, 2),\n'
    '        "shortfall_per_hour": round(shortfall_per_hour, 2),\n'
    '        "covariance_share_of_mean_revenue_pct": round(\n'
    "            covariance_share_of_level, 3\n"
    "        ),\n",
    "cell 32 covariance json",
)
set_source(cells[32], source)

# --------------------------------------------------------------------------
# Markdown 33 — the two stale claims, and the new guidance
# --------------------------------------------------------------------------
source = cell_source(cells[33])
source = sub(
    source,
    "(overdispersion, independence), the quasi-Poisson correction, and the "
    "negative binomial\nagreement.",
    "(overdispersion, independence), the quasi-Poisson correction, and what "
    "the negative\nbinomial contrast actually showed — read the printed "
    "verdict rather than assuming\nagreement, and where the two disagree, say "
    "which loss each wins and why the Poisson is the\none that carries the "
    "recommendations.",
    "md 33 stale nb agreement",
)
source = sub(
    source,
    "one that carries the recommendations. For Model 2: why a contrasting "
    "family rather than a second GLM, and that the\nhyper-parameters were "
    "chosen on a forward validation split.",
    "one that carries the recommendations. For Model 2: why a contrasting "
    "family rather than\na second GLM, and that the hyper-parameters were "
    "chosen on a forward validation\nsplit.",
    "md 33 rewrap",
)
source = sub(
    source,
    "**Report the error analysis, including where the model is biased and what "
    "the bias is.**",
    "**State what the product cannot represent.** `E[pickups] × E[fare]` is "
    "not `E[revenue]`\nunless the two are conditionally uncorrelated, and the "
    "gap is exactly\n`Cov(pickups, fare | X)`, which Step 6 measures and "
    "prints against the shortfall it is\nmeant to explain. One sentence, and "
    "it belongs in the modelling section rather than the\nlimitations: it is a "
    "property of combining two conditional means, not a defect of either\nfit. "
    "Quoting it is the difference between a combination that was checked and "
    "one that was\nassumed.\n\n"
    "**Report the error analysis, including where the model is biased and what "
    "the bias is.**",
    "md 33 covariance guidance",
)
set_source(cells[33], source)

NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print("notebook patched")

# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------
raw = README.read_text()
crlf = "\r\n" in raw
text = raw.replace("\r\n", "\n")

text = sub(
    text,
    "A negative binomial is fitted on the same design as a contrast, with its "
    "dispersion\nestimated by the standard auxiliary regression, so the demand "
    "result can be reported as\nrobust to the variance assumption rather than "
    "conditional on it.",
    "A negative binomial is fitted on the same design as a contrast, with its "
    "dispersion\nestimated by the standard auxiliary regression, so the demand "
    "result's sensitivity to the\nvariance assumption is measured rather than "
    "assumed. The two are compared on squared\nerror and on mean Poisson "
    "deviance, since squared error alone favours the Poisson by\n"
    "construction — a negative binomial's IRLS weight discounts exactly the "
    "busy hours where\nsquared error is decided.",
    "readme nb robustness",
)

text = sub(
    text,
    "combination is evaluable on every test hour, including the empty ones "
    "where the value\nmodel alone is undefined, against the product of the two "
    "seasonal-naive forecasts.",
    "combination is evaluable on every test hour, including the empty ones "
    "where the value\nmodel alone is undefined, against the product of the two "
    "seasonal-naive forecasts. That\nidentity holds row by row and not in "
    "expectation — a product of two conditional means\nomits "
    "`Cov(n_pickups, mean_total | X)` — so notebook 4 measures the omitted "
    "term and\nreports it beside the combination's error rather than charging "
    "the whole shortfall to the\ndemand model.",
    "readme covariance",
)

if crlf:
    text = text.replace("\n", "\r\n")
README.write_text(text)
print("readme patched")
