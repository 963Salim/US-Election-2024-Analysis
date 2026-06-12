# ============================================================
# US Election 2024 - Logistic Regression Analysis
# Python version of the R project
# ============================================================

from __future__ import annotations

import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score


# ------------------------------------------------------------
# 0. Settings
# ------------------------------------------------------------

RANDOM_STATE = 123

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

def clean_column_name(name: str) -> str:
    """Convert column names to lower snake_case."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def find_data_file() -> Path:
    possible_paths = [
        Path("data/US_election_2024.csv"),
        Path("US_election_2024.csv"),
        Path("data/US_election_2024_cleaned.csv"),
        Path("US_election_2024_cleaned.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Keine CSV-Datei gefunden. Lege US_election_2024.csv oder "
        "US_election_2024_cleaned.csv entweder im Projektordner oder im Ordner data/ ab."
    )


def parse_number(value):
    """Robust numeric parser for decimal comma and decimal point formats."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return value

    x = str(value).strip().replace(" ", "")

    if x == "":
        return np.nan

    # Case: German style 1.234,56
    if "," in x and "." in x and x.rfind(",") > x.rfind("."):
        x = x.replace(".", "").replace(",", ".")

    # Case: US style 1,234.56
    elif "," in x and "." in x and x.rfind(".") > x.rfind(","):
        x = x.replace(",", "")

    # Case: decimal comma 12,34
    elif "," in x:
        x = x.replace(",", ".")

    return pd.to_numeric(x, errors="coerce")


def as_percent(series: pd.Series) -> pd.Series:
    """
    Converts rates to percentage points.
    If values are between 0 and 1, multiply by 100.
    If values are already like 2.6, 5.5, keep them.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.max(skipna=True) <= 1:
        return numeric * 100
    return numeric


def fit_logit_model(data: pd.DataFrame, response: str, predictors: list[str]):
    """Fit a logistic regression model with statsmodels GLM."""
    X = data[predictors].copy()
    X = sm.add_constant(X, has_constant="add")
    y = data[response]

    model = sm.GLM(y, X, family=sm.families.Binomial())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(maxiter=200)

    return result


def create_coefficient_table(model_result) -> pd.DataFrame:
    """Create coefficient table with confidence intervals and odds ratios."""
    params = model_result.params
    bse = model_result.bse
    z_values = params / bse
    p_values = model_result.pvalues
    conf_int = model_result.conf_int()

    table = pd.DataFrame({
        "Variable": params.index,
        "Estimate": params.values,
        "Std_Error": bse.values,
        "z_value": z_values.values,
        "p_value": p_values.values,
        "CI_low": conf_int[0].values,
        "CI_high": conf_int[1].values,
    })

    table["Odds_Ratio"] = np.exp(table["Estimate"])
    table["OR_CI_low"] = np.exp(table["CI_low"])
    table["OR_CI_high"] = np.exp(table["CI_high"])

    numeric_cols = table.select_dtypes(include=[np.number]).columns
    table[numeric_cols] = table[numeric_cols].round(4)

    return table


def best_subset_selection(
    data: pd.DataFrame,
    response: str,
    predictors: list[str],
    criterion: str = "AIC",
) -> tuple[pd.DataFrame, object, list[str]]:
    """
    Test all predictor subsets and select the best model by AIC or BIC.
    """
    if criterion not in {"AIC", "BIC"}:
        raise ValueError("criterion must be either 'AIC' or 'BIC'.")

    rows = []
    best_score = np.inf
    best_model = None
    best_predictors = None

    all_subsets = [[]]

    for k in range(1, len(predictors) + 1):
        all_subsets.extend(list(itertools.combinations(predictors, k)))

    for subset in all_subsets:
        subset = list(subset)

        try:
            if len(subset) == 0:
                X = pd.DataFrame({"const": 1}, index=data.index)
                y = data[response]
                model = sm.GLM(y, X, family=sm.families.Binomial())

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(maxiter=200)
            else:
                result = fit_logit_model(data, response, subset)

            aic = result.aic
            bic = result.bic

            model_name = "Intercept only" if len(subset) == 0 else " + ".join(subset)

            rows.append({
                "Model": model_name,
                "n_predictors": len(subset),
                "AIC": aic,
                "BIC": bic,
            })

            score = aic if criterion == "AIC" else bic

            if score < best_score:
                best_score = score
                best_model = result
                best_predictors = subset

        except Exception as error:
            rows.append({
                "Model": " + ".join(subset) if subset else "Intercept only",
                "n_predictors": len(subset),
                "AIC": np.nan,
                "BIC": np.nan,
                "error": str(error),
            })

    selection_table = pd.DataFrame(rows)
    selection_table = selection_table.sort_values(criterion, na_position="last").reset_index(drop=True)

    return selection_table, best_model, best_predictors


def make_logistic_pipeline() -> Pipeline:
    """
    Logistic regression pipeline for cross-validation.
    Scaling is done inside each CV fold to avoid data leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            random_state=RANDOM_STATE,
        )),
    ])


# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------

data_path = find_data_file()
print(f"Datensatz wird eingelesen aus: {data_path}")

df_raw = pd.read_csv(data_path, sep=None, engine="python", dtype=str)
df_raw.columns = [clean_column_name(col) for col in df_raw.columns]

required_columns = [
    "state",
    "leading_candidate",
    "total_area",
    "population",
    "population_density",
    "median_age",
    "birth_rate",
    "hdi",
    "unemployment_rate",
    "health_insurance_coverage",
    "median_rent",
]

missing_columns = [col for col in required_columns if col not in df_raw.columns]

if missing_columns:
    raise ValueError(f"Folgende benötigte Spalten fehlen: {missing_columns}")


# ------------------------------------------------------------
# 3. Data preparation
# ------------------------------------------------------------

numeric_variables = [
    "total_area",
    "population",
    "population_density",
    "median_age",
    "birth_rate",
    "hdi",
    "unemployment_rate",
    "health_insurance_coverage",
    "median_rent",
]

df = df_raw.copy()

for col in numeric_variables:
    df[col] = df[col].apply(parse_number)

df["state"] = df["state"].str.lower().str.strip()
df["leading_candidate"] = df["leading_candidate"].str.strip()

# Target variable:
# Harris = 1, Trump = 0
df["target_harris"] = np.where(df["leading_candidate"] == "Harris", 1, 0)

# Transform skewed variables
df["log_total_area"] = np.log(df["total_area"])
df["log_population"] = np.log(df["population"])

# Convert rates to percentage points if needed
df["unemployment_rate_pct"] = as_percent(df["unemployment_rate"])
df["health_insurance_coverage_pct"] = as_percent(df["health_insurance_coverage"])

if df.isna().any().any():
    print("\nFehlende Werte nach Datenaufbereitung:")
    print(df.isna().sum()[df.isna().sum() > 0])
    raise ValueError("Der aufbereitete Datensatz enthält fehlende Werte.")

print("\nAnzahl Beobachtungen:", len(df))
print("\nVerteilung der Zielvariable:")
print(df["leading_candidate"].value_counts())


# ------------------------------------------------------------
# 4. Descriptive analysis
# ------------------------------------------------------------

descriptive_variables = [
    "total_area",
    "population",
    "population_density",
    "median_age",
    "birth_rate",
    "hdi",
    "unemployment_rate_pct",
    "health_insurance_coverage_pct",
    "median_rent",
]

descriptive_table = df[descriptive_variables].agg([
    "mean",
    "median",
    "std",
    "min",
    "max",
]).T

descriptive_table.columns = [
    "Mittelwert",
    "Median",
    "SD",
    "Minimum",
    "Maximum",
]

descriptive_table = descriptive_table.round(4)
descriptive_table.to_csv(TABLE_DIR / "descriptive_statistics.csv")

print("\nDeskriptive Statistiken:")
print(descriptive_table)


# ------------------------------------------------------------
# 5. Descriptive plots
# ------------------------------------------------------------

candidate_colors = {
    "Harris": "#2563EB",
    "Trump": "#DC2626",
}

# 5.1 Bar chart
counts = df["leading_candidate"].value_counts().reindex(["Harris", "Trump"])

plt.figure(figsize=(7, 5))
plt.bar(counts.index, counts.values, color=[candidate_colors[x] for x in counts.index], edgecolor="black")
plt.ylim(0, 50)
plt.yticks(range(0, 51, 5))
plt.xlabel("Kandidat")
plt.ylabel("Anzahl der Staaten")
plt.title("Verteilung der gewonnenen Staaten")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_candidate_counts.pdf")
plt.close()

# 5.2 Histograms
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
axes = axes.ravel()

for ax, col in zip(axes, descriptive_variables):
    ax.hist(df[col], bins=10, edgecolor="black")
    ax.set_title(col)
    ax.set_ylabel("Häufigkeit")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_histograms.pdf")
plt.close()

# 5.3 Boxplots by candidate
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
axes = axes.ravel()

for ax, col in zip(axes, descriptive_variables):
    data_harris = df.loc[df["leading_candidate"] == "Harris", col]
    data_trump = df.loc[df["leading_candidate"] == "Trump", col]

    ax.boxplot([data_harris, data_trump], labels=["Harris", "Trump"])
    ax.set_title(col)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_boxplots_by_candidate.pdf")
plt.close()

# 5.4 Scatterplot: insurance vs rent
plt.figure(figsize=(7, 5))

for candidate in ["Harris", "Trump"]:
    subset = df[df["leading_candidate"] == candidate]
    plt.scatter(
        subset["health_insurance_coverage_pct"],
        subset["median_rent"],
        label=candidate,
        alpha=0.85,
        s=50,
        color=candidate_colors[candidate],
    )

plt.xlabel("Versicherungsrate (%)")
plt.ylabel("Mediane Miete (US-Dollar)")
plt.title("Versicherungsrate und mediane Miete nach Wahlausgang")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_scatter_insurance_rent.pdf")
plt.close()


# ------------------------------------------------------------
# 6. Model variables
# ------------------------------------------------------------

# population_density is excluded because it is structurally derived from
# population and total_area.
model_predictors = [
    "log_total_area",
    "log_population",
    "median_age",
    "birth_rate",
    "hdi",
    "unemployment_rate_pct",
    "health_insurance_coverage_pct",
    "median_rent",
]

response = "target_harris"


# ------------------------------------------------------------
# 7. Multicollinearity diagnostics
# ------------------------------------------------------------

correlation_matrix = df[model_predictors].corr().round(3)
correlation_matrix.to_csv(TABLE_DIR / "correlation_matrix.csv")

print("\nKorrelationsmatrix:")
print(correlation_matrix)


def calculate_vif(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    rows = []

    for variable in predictors:
        other_variables = [v for v in predictors if v != variable]

        X = sm.add_constant(data[other_variables], has_constant="add")
        y = data[variable]

        result = sm.OLS(y, X).fit()
        r_squared = result.rsquared

        vif = 1 / (1 - r_squared)

        rows.append({
            "Variable": variable,
            "VIF": vif,
        })

    return pd.DataFrame(rows)


vif_table = calculate_vif(df, model_predictors)
vif_table["VIF"] = vif_table["VIF"].round(4)
vif_table.to_csv(TABLE_DIR / "vif_table.csv", index=False)

print("\nVIF-Werte:")
print(vif_table)


# ------------------------------------------------------------
# 8. Full logistic regression model
# ------------------------------------------------------------

full_model = fit_logit_model(df, response, model_predictors)

print("\nVollständiges logistisches Modell:")
print(full_model.summary())

print("\nAIC vollständiges Modell:", round(full_model.aic, 3))
print("BIC vollständiges Modell:", round(full_model.bic, 3))

full_coefficients = create_coefficient_table(full_model)
full_coefficients.to_csv(TABLE_DIR / "full_model_coefficients.csv", index=False)

print("\nKoeffiziententabelle vollständiges Modell:")
print(full_coefficients)


# ------------------------------------------------------------
# 9. Best-subset selection by AIC
# ------------------------------------------------------------

selection_table, reduced_model, reduced_predictors = best_subset_selection(
    data=df,
    response=response,
    predictors=model_predictors,
    criterion="AIC",
)

selection_table.to_csv(TABLE_DIR / "model_selection_aic.csv", index=False)

print("\nBeste Modelle nach AIC:")
print(selection_table.head(10))

print("\nAusgewähltes reduziertes Modell:")
print("target_harris ~ " + " + ".join(reduced_predictors))
print("AIC reduziertes Modell:", round(reduced_model.aic, 3))
print("BIC reduziertes Modell:", round(reduced_model.bic, 3))

print("\nReduziertes Modell:")
print(reduced_model.summary())

reduced_coefficients = create_coefficient_table(reduced_model)
reduced_coefficients.to_csv(TABLE_DIR / "reduced_model_coefficients.csv", index=False)

print("\nKoeffiziententabelle reduziertes Modell:")
print(reduced_coefficients)


# ------------------------------------------------------------
# 10. Odds-ratio plot for reduced model
# ------------------------------------------------------------

or_data = reduced_coefficients[reduced_coefficients["Variable"] != "const"].copy()

or_data = or_data.replace({
    "Variable": {
        "log_total_area": "log(Gesamtfläche)",
        "log_population": "log(Einwohnerzahl)",
        "median_age": "Medianes Alter",
        "birth_rate": "Geburtenrate",
        "hdi": "HDI",
        "unemployment_rate_pct": "Arbeitslosenrate (%)",
        "health_insurance_coverage_pct": "Versicherungsrate (%)",
        "median_rent": "Mediane Miete",
    }
})

or_data = or_data.sort_values("Odds_Ratio")

plt.figure(figsize=(8, 5))
plt.errorbar(
    x=or_data["Odds_Ratio"],
    y=or_data["Variable"],
    xerr=[
        or_data["Odds_Ratio"] - or_data["OR_CI_low"],
        or_data["OR_CI_high"] - or_data["Odds_Ratio"],
    ],
    fmt="o",
    capsize=4,
)

plt.axvline(1, linestyle="--")
plt.xscale("log")
plt.xlabel("Odds Ratio mit 95%-Wald-Konfidenzintervall")
plt.ylabel("")
plt.title("Odds Ratios des reduzierten Modells")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_odds_ratios_reduced_model.pdf")
plt.close()


# ------------------------------------------------------------
# 11. Cross-validation and ROC/AUC
# ------------------------------------------------------------

X_full = df[model_predictors]
X_reduced = df[reduced_predictors]
y = df[response]

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE,
)

full_pipeline = make_logistic_pipeline()
reduced_pipeline = make_logistic_pipeline()

full_probs = cross_val_predict(
    full_pipeline,
    X_full,
    y,
    cv=cv,
    method="predict_proba",
)[:, 1]

reduced_probs = cross_val_predict(
    reduced_pipeline,
    X_reduced,
    y,
    cv=cv,
    method="predict_proba",
)[:, 1]

auc_full = roc_auc_score(y, full_probs)
auc_reduced = roc_auc_score(y, reduced_probs)

print("\nCross-Validated AUC vollständiges Modell:", round(auc_full, 3))
print("Cross-Validated AUC reduziertes Modell:", round(auc_reduced, 3))

auc_table = pd.DataFrame({
    "Modell": ["Vollständiges Modell", "Reduziertes Modell"],
    "AUC": [auc_full, auc_reduced],
})

auc_table.to_csv(TABLE_DIR / "cross_validated_auc.csv", index=False)

fpr_full, tpr_full, _ = roc_curve(y, full_probs)
fpr_reduced, tpr_reduced, _ = roc_curve(y, reduced_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_full, tpr_full, linestyle="--", label=f"Vollständiges Modell AUC: {auc_full:.3f}")
plt.plot(fpr_reduced, tpr_reduced, linestyle="-", label=f"Reduziertes Modell AUC: {auc_reduced:.3f}")
plt.plot([0, 1], [0, 1], linestyle=":", color="gray")

plt.xlabel("Falsch-Positiv-Rate (1 - Spezifität)")
plt.ylabel("Sensitivität (Wahre-Positiv-Rate)")
plt.title("Cross-Validated ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_cross_validated_roc.pdf")
plt.close()


# ------------------------------------------------------------
# 12. Final summary
# ------------------------------------------------------------

significant_variables = reduced_coefficients[
    (reduced_coefficients["Variable"] != "const")
    & (reduced_coefficients["p_value"] < 0.05)
]

print("\n============================================================")
print("ERGEBNISZUSAMMENFASSUNG")
print("============================================================")

print("\nZielvariable:")
print("Harris = 1, Trump = 0")

print("\nVollständiges Modell:")
print("Prädiktoren:", ", ".join(model_predictors))
print("AIC:", round(full_model.aic, 3))
print("CV-AUC:", round(auc_full, 3))

print("\nReduziertes Modell:")
print("Prädiktoren:", ", ".join(reduced_predictors))
print("AIC:", round(reduced_model.aic, 3))
print("CV-AUC:", round(auc_reduced, 3))

print("\nSignifikante Variablen im reduzierten Modell auf 5%-Niveau:")
print(significant_variables)

print("\nHinweis zur Interpretation:")
print(
    "Die Ergebnisse beschreiben Assoziationen auf Bundesstaatenebene. "
    "Sie erlauben keine kausalen Aussagen über individuelles Wahlverhalten."
)

print("\nAlle Tabellen und Grafiken wurden im Ordner outputs/ gespeichert.")