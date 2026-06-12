# ============================================================
# Projekt 5: US Election 2024 - Logistische Regression
# Bereinigte und korrigierte Version
# ============================================================

# ------------------------------------------------------------
# 0. Pakete laden
# ------------------------------------------------------------

required_packages <- c(
  "tidyverse",
  "janitor",
  "caret",
  "pROC",
  "ggplot2",
  "patchwork",
  "usmap"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, dependencies = TRUE)
}

invisible(lapply(required_packages, library, character.only = TRUE))

set.seed(123)


# ------------------------------------------------------------
# 1. Ordnerstruktur für Ergebnisse
# ------------------------------------------------------------

dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------
# 2. Datensatz einlesen
# ------------------------------------------------------------

find_data_file <- function() {
  possible_paths <- c(
    "data/US_election_2024.csv",
    "US_election_2024.csv",
    "data/US_election_2024_cleaned.csv",
    "US_election_2024_cleaned.csv"
  )
  
  existing_paths <- possible_paths[file.exists(possible_paths)]
  
  if (length(existing_paths) == 0) {
    stop(
      "Keine CSV-Datei gefunden. Lege US_election_2024.csv entweder ",
      "im Projektordner oder im Ordner data/ ab."
    )
  }
  
  existing_paths[1]
}


read_election_data <- function(file_path) {
  first_line <- readLines(file_path, n = 1, warn = FALSE)
  
  if (grepl(";", first_line)) {
    data <- readr::read_delim(
      file = file_path,
      delim = ";",
      locale = readr::locale(decimal_mark = ",", grouping_mark = "."),
      show_col_types = FALSE,
      trim_ws = TRUE
    )
  } else {
    data <- readr::read_csv(
      file = file_path,
      locale = readr::locale(decimal_mark = ",", grouping_mark = "."),
      show_col_types = FALSE,
      trim_ws = TRUE
    )
  }
  
  data
}


convert_to_numeric <- function(x) {
  if (is.numeric(x)) {
    return(x)
  }
  
  readr::parse_number(
    as.character(x),
    locale = readr::locale(decimal_mark = ",", grouping_mark = ".")
  )
}


data_path <- find_data_file()
cat("Datensatz wird eingelesen aus:", data_path, "\n")

df_raw <- read_election_data(data_path) |>
  janitor::clean_names()

required_columns <- c(
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
  "median_rent"
)

missing_columns <- setdiff(required_columns, names(df_raw))

if (length(missing_columns) > 0) {
  stop(
    "Folgende benötigte Spalten fehlen im Datensatz: ",
    paste(missing_columns, collapse = ", ")
  )
}


# ------------------------------------------------------------
# 3. Datenaufbereitung
# ------------------------------------------------------------

numeric_variables <- c(
  "total_area",
  "population",
  "population_density",
  "median_age",
  "birth_rate",
  "hdi",
  "unemployment_rate",
  "health_insurance_coverage",
  "median_rent"
)

df <- df_raw |>
  mutate(
    across(all_of(numeric_variables), convert_to_numeric),
    state = tolower(state),
    leading_candidate = factor(
      leading_candidate,
      levels = c("Harris", "Trump")
    ),
    
    # Zielvariable eindeutig:
    # Harris = 1, Trump = 0
    target_harris = if_else(leading_candidate == "Harris", 1L, 0L),
    
    # Für caret:
    # Harris ist die positive Klasse und steht deshalb zuerst.
    candidate_factor = factor(
      leading_candidate,
      levels = c("Harris", "Trump")
    ),
    
    # Transformation schiefer Variablen
    log_total_area = log(total_area),
    log_population = log(population),
    
    # Raten als Prozentpunkte, damit die Koeffizienten interpretierbarer sind
    unemployment_rate_pct = unemployment_rate * 100,
    health_insurance_coverage_pct = health_insurance_coverage * 100
  )

if (anyNA(df)) {
  stop("Der aufbereitete Datensatz enthält fehlende Werte. Bitte Daten prüfen.")
}

cat("Anzahl Beobachtungen:", nrow(df), "\n")
cat("Verteilung der Zielvariable:\n")
print(table(df$leading_candidate))


# ------------------------------------------------------------
# 4. Deskriptive Analyse
# ------------------------------------------------------------

descriptive_variables <- c(
  "total_area",
  "population",
  "population_density",
  "median_age",
  "birth_rate",
  "hdi",
  "unemployment_rate_pct",
  "health_insurance_coverage_pct",
  "median_rent"
)

variable_labels <- c(
  total_area = "Gesamtfläche",
  population = "Einwohnerzahl",
  population_density = "Bevölkerungsdichte",
  median_age = "Medianes Alter",
  birth_rate = "Geburtenrate",
  hdi = "HDI",
  unemployment_rate_pct = "Arbeitslosenrate (%)",
  health_insurance_coverage_pct = "Versicherungsrate (%)",
  median_rent = "Mediane Miete"
)

descriptive_table <- df |>
  summarise(
    across(
      all_of(descriptive_variables),
      list(
        Mittelwert = ~ mean(.x),
        Median = ~ median(.x),
        SD = ~ sd(.x),
        Minimum = ~ min(.x),
        Maximum = ~ max(.x)
      ),
      .names = "{.col}_{.fn}"
    )
  )

write_csv(descriptive_table, "outputs/tables/descriptive_statistics.csv")

cat("\nDeskriptive Statistiken:\n")
print(descriptive_table)


# ------------------------------------------------------------
# 5. Deskriptive Grafiken
# ------------------------------------------------------------

# 5.1 Balkendiagramm: Harris vs. Trump
plot_candidate_counts <- ggplot(df, aes(x = leading_candidate, fill = leading_candidate)) +
  geom_bar(color = "black", linewidth = 0.3) +
  scale_fill_manual(
    values = c("Harris" = "#2563EB", "Trump" = "#DC2626"),
    name = "Kandidat"
  ) +
  scale_y_continuous(breaks = seq(0, 50, 5), limits = c(0, 50)) +
  labs(
    x = "Kandidat",
    y = "Anzahl der Staaten"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

print(plot_candidate_counts)

ggsave(
  filename = "outputs/figures/01_candidate_counts.pdf",
  plot = plot_candidate_counts,
  width = 7,
  height = 5
)


# 5.2 Histogramme der numerischen Variablen
df_long <- df |>
  select(all_of(descriptive_variables)) |>
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  ) |>
  mutate(
    variable = recode(variable, !!!variable_labels)
  )

plot_histograms <- ggplot(df_long, aes(x = value)) +
  geom_histogram(bins = 10, fill = "lightblue", color = "black") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  labs(
    x = NULL,
    y = "Häufigkeit"
  ) +
  theme_minimal(base_size = 12)

print(plot_histograms)

ggsave(
  filename = "outputs/figures/02_histograms.pdf",
  plot = plot_histograms,
  width = 10,
  height = 8
)


# 5.3 Stratifizierte Boxplots nach Wahlergebnis
# Diese Grafik ist wichtig, weil sie Unterschiede zwischen Harris- und Trump-Staaten zeigt.
df_boxplots <- df |>
  select(candidate_factor, all_of(descriptive_variables)) |>
  pivot_longer(
    cols = -candidate_factor,
    names_to = "variable",
    values_to = "value"
  ) |>
  mutate(
    variable = recode(variable, !!!variable_labels)
  )

plot_boxplots <- ggplot(
  df_boxplots,
  aes(x = candidate_factor, y = value, fill = candidate_factor)
) +
  geom_boxplot(alpha = 0.75, outlier.alpha = 0.8) +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  scale_fill_manual(
    values = c("Harris" = "#2563EB", "Trump" = "#DC2626"),
    name = "Kandidat"
  ) +
  labs(
    x = "Gewinner des Bundesstaates",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom"
  )

print(plot_boxplots)

ggsave(
  filename = "outputs/figures/03_boxplots_by_candidate.pdf",
  plot = plot_boxplots,
  width = 10,
  height = 8
)


# 5.4 Scatterplot: Versicherungsrate vs. mediane Miete
plot_scatter <- ggplot(
  df,
  aes(
    x = health_insurance_coverage_pct,
    y = median_rent,
    color = leading_candidate
  )
) +
  geom_point(size = 3, alpha = 0.85) +
  scale_color_manual(
    values = c("Harris" = "#2563EB", "Trump" = "#DC2626"),
    name = "Kandidat"
  ) +
  labs(
    x = "Versicherungsrate (%)",
    y = "Mediane Miete (US-Dollar)"
  ) +
  theme_minimal(base_size = 14)

print(plot_scatter)

ggsave(
  filename = "outputs/figures/04_scatter_insurance_rent.pdf",
  plot = plot_scatter,
  width = 7,
  height = 5
)


# 5.5 US-Karte nach Wahlergebnis
plot_map <- usmap::plot_usmap(
  data = df,
  values = "leading_candidate",
  color = "white"
) +
  scale_fill_manual(
    values = c("Harris" = "#2563EB", "Trump" = "#DC2626"),
    name = "Kandidat"
  ) +
  labs(title = "US Election 2024: Gewinner nach Bundesstaat") +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold")
  )

print(plot_map)

ggsave(
  filename = "outputs/figures/05_us_map_candidate.pdf",
  plot = plot_map,
  width = 9,
  height = 6
)


# ------------------------------------------------------------
# 6. Modellvariablen definieren
# ------------------------------------------------------------

# Population_Density wird nicht im Hauptmodell verwendet,
# weil sie direkt aus Population und Total_Area abgeleitet ist.
# Dadurch würde starke strukturelle Abhängigkeit entstehen.

model_predictors <- c(
  "log_total_area",
  "log_population",
  "median_age",
  "birth_rate",
  "hdi",
  "unemployment_rate_pct",
  "health_insurance_coverage_pct",
  "median_rent"
)

full_formula <- as.formula(
  paste("target_harris ~", paste(model_predictors, collapse = " + "))
)


# ------------------------------------------------------------
# 7. Multikollinearität grob prüfen
# ------------------------------------------------------------

correlation_matrix <- cor(df[, model_predictors])

write.csv(
  round(correlation_matrix, 3),
  "outputs/tables/correlation_matrix.csv"
)

cat("\nKorrelationsmatrix der Modellvariablen:\n")
print(round(correlation_matrix, 3))


calculate_vif <- function(data, predictors) {
  vif_values <- sapply(predictors, function(variable) {
    other_variables <- setdiff(predictors, variable)
    
    formula <- as.formula(
      paste(variable, "~", paste(other_variables, collapse = " + "))
    )
    
    r_squared <- summary(lm(formula, data = data))$r.squared
    1 / (1 - r_squared)
  })
  
  data.frame(
    Variable = names(vif_values),
    VIF = as.numeric(vif_values),
    row.names = NULL
  )
}

vif_table <- calculate_vif(df, model_predictors)

write_csv(vif_table, "outputs/tables/vif_table.csv")

cat("\nVIF-Werte:\n")
print(vif_table)


# ------------------------------------------------------------
# 8. Vollständiges logistisches Modell
# ------------------------------------------------------------

full_model <- glm(
  formula = full_formula,
  data = df,
  family = binomial(link = "logit")
)

cat("\nVollständiges logistisches Modell:\n")
print(summary(full_model))

cat("\nAIC vollständiges Modell:", AIC(full_model), "\n")
cat("BIC vollständiges Modell:", BIC(full_model), "\n")


# ------------------------------------------------------------
# 9. Funktion für Koeffiziententabelle
# ------------------------------------------------------------

create_coefficient_table <- function(model) {
  coefficient_matrix <- summary(model)$coefficients
  confidence_intervals <- confint.default(model)
  
  table <- data.frame(
    Variable = rownames(coefficient_matrix),
    Estimate = coefficient_matrix[, "Estimate"],
    Std_Error = coefficient_matrix[, "Std. Error"],
    z_value = coefficient_matrix[, "z value"],
    p_value = coefficient_matrix[, "Pr(>|z|)"],
    CI_low = confidence_intervals[, 1],
    CI_high = confidence_intervals[, 2],
    Odds_Ratio = exp(coefficient_matrix[, "Estimate"]),
    OR_CI_low = exp(confidence_intervals[, 1]),
    OR_CI_high = exp(confidence_intervals[, 2]),
    row.names = NULL
  )
  
  table |>
    mutate(
      across(
        where(is.numeric),
        ~ round(.x, 4)
      )
    )
}

full_model_coefficients <- create_coefficient_table(full_model)

write_csv(
  full_model_coefficients,
  "outputs/tables/full_model_coefficients.csv"
)

cat("\nKoeffiziententabelle vollständiges Modell:\n")
print(full_model_coefficients)


# ------------------------------------------------------------
# 10. Best-Subset-Selection mit AIC
# ------------------------------------------------------------

fit_all_subsets <- function(data, response, predictors, criterion = "AIC") {
  if (!criterion %in% c("AIC", "BIC")) {
    stop("criterion muss entweder 'AIC' oder 'BIC' sein.")
  }
  
  predictor_subsets <- list(character(0))
  
  for (k in seq_along(predictors)) {
    predictor_subsets <- c(
      predictor_subsets,
      combn(predictors, k, simplify = FALSE)
    )
  }
  
  results <- vector("list", length(predictor_subsets))
  
  for (i in seq_along(predictor_subsets)) {
    subset <- predictor_subsets[[i]]
    
    rhs <- if (length(subset) == 0) {
      "1"
    } else {
      paste(subset, collapse = " + ")
    }
    
    formula_text <- paste(response, "~", rhs)
    model_formula <- as.formula(formula_text)
    
    model <- glm(
      formula = model_formula,
      data = data,
      family = binomial(link = "logit")
    )
    
    results[[i]] <- list(
      model_name = rhs,
      formula = formula_text,
      n_predictors = length(subset),
      model = model,
      AIC = AIC(model),
      BIC = BIC(model)
    )
  }
  
  selection_table <- bind_rows(
    lapply(results, function(result) {
      tibble(
        Model = result$model_name,
        Formula = result$formula,
        n_predictors = result$n_predictors,
        AIC = result$AIC,
        BIC = result$BIC
      )
    })
  ) |>
    arrange(.data[[criterion]])
  
  best_index <- which.min(
    sapply(results, function(result) result[[criterion]])
  )
  
  list(
    all_results = results,
    selection_table = selection_table,
    best_model = results[[best_index]]$model,
    best_formula = results[[best_index]]$formula,
    best_criterion_value = results[[best_index]][[criterion]],
    criterion = criterion
  )
}

selection_result <- fit_all_subsets(
  data = df,
  response = "target_harris",
  predictors = model_predictors,
  criterion = "AIC"
)

selection_table <- selection_result$selection_table

write_csv(
  selection_table,
  "outputs/tables/model_selection_aic.csv"
)

cat("\nBeste Modelle nach AIC:\n")
print(head(selection_table, 10))

cat("\nAusgewähltes reduziertes Modell:\n")
cat(selection_result$best_formula, "\n")
cat("AIC reduziertes Modell:", AIC(selection_result$best_model), "\n")
cat("BIC reduziertes Modell:", BIC(selection_result$best_model), "\n")

reduced_model <- selection_result$best_model

cat("\nReduziertes Modell:\n")
print(summary(reduced_model))

reduced_model_coefficients <- create_coefficient_table(reduced_model)

write_csv(
  reduced_model_coefficients,
  "outputs/tables/reduced_model_coefficients.csv"
)

cat("\nKoeffiziententabelle reduziertes Modell:\n")
print(reduced_model_coefficients)


# ------------------------------------------------------------
# 11. Odds-Ratio-Plot für reduziertes Modell
# ------------------------------------------------------------

or_plot_data <- reduced_model_coefficients |>
  filter(Variable != "(Intercept)") |>
  mutate(
    Variable = recode(
      Variable,
      log_total_area = "log(Gesamtfläche)",
      log_population = "log(Einwohnerzahl)",
      median_age = "Medianes Alter",
      birth_rate = "Geburtenrate",
      hdi = "HDI",
      unemployment_rate_pct = "Arbeitslosenrate (%)",
      health_insurance_coverage_pct = "Versicherungsrate (%)",
      median_rent = "Mediane Miete"
    )
  )

plot_odds_ratios <- ggplot(
  or_plot_data,
  aes(x = Odds_Ratio, y = reorder(Variable, Odds_Ratio))
) +
  geom_point(size = 3) +
  geom_errorbarh(
    aes(xmin = OR_CI_low, xmax = OR_CI_high),
    height = 0.2
  ) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  scale_x_log10() +
  labs(
    x = "Odds Ratio mit 95%-Wald-Konfidenzintervall",
    y = NULL
  ) +
  theme_minimal(base_size = 13)

print(plot_odds_ratios)

ggsave(
  filename = "outputs/figures/06_odds_ratios_reduced_model.pdf",
  plot = plot_odds_ratios,
  width = 8,
  height = 5
)


# ------------------------------------------------------------
# 12. Kreuzvalidierung für vollständiges und reduziertes Modell
# ------------------------------------------------------------

set.seed(123)

# Zielvariable explizit prüfen
df$candidate_factor <- factor(
  df$candidate_factor,
  levels = c("Harris", "Trump")
)

cat("\nLevels der Zielvariable für caret:\n")
print(levels(df$candidate_factor))
print(table(df$candidate_factor))

# Folds erstellen
folds <- caret::createFolds(
  y = df$candidate_factor,
  k = 10,
  returnTrain = TRUE
)

cv_control <- caret::trainControl(
  method = "cv",
  number = 10,
  index = folds,
  classProbs = TRUE,
  summaryFunction = caret::twoClassSummary,
  savePredictions = "final"
)

# Prädiktoren für vollständiges Modell
x_full <- df[, model_predictors]
y <- df$candidate_factor

# Prädiktoren für reduziertes Modell
best_predictors <- attr(terms(reduced_model), "term.labels")
x_reduced <- df[, best_predictors]

cat("\nPrädiktoren vollständiges Modell:\n")
print(names(x_full))

cat("\nPrädiktoren reduziertes Modell:\n")
print(names(x_reduced))

set.seed(123)
cv_full_model <- caret::train(
  x = x_full,
  y = y,
  method = "glm",
  family = binomial(link = "logit"),
  trControl = cv_control,
  metric = "ROC"
)

set.seed(123)
cv_reduced_model <- caret::train(
  x = x_reduced,
  y = y,
  method = "glm",
  family = binomial(link = "logit"),
  trControl = cv_control,
  metric = "ROC"
)

cat("\nCaret-Ergebnisse vollständiges Modell:\n")
print(cv_full_model)

cat("\nCaret-Ergebnisse reduziertes Modell:\n")
print(cv_reduced_model)


# ------------------------------------------------------------
# 13. ROC-Kurven und AUC aus Kreuzvalidierung
# ------------------------------------------------------------

roc_full_cv <- pROC::roc(
  response = cv_full_model$pred$obs,
  predictor = cv_full_model$pred$Harris,
  levels = c("Trump", "Harris"),
  direction = "<"
)

roc_reduced_cv <- pROC::roc(
  response = cv_reduced_model$pred$obs,
  predictor = cv_reduced_model$pred$Harris,
  levels = c("Trump", "Harris"),
  direction = "<"
)

auc_full_cv <- pROC::auc(roc_full_cv)
auc_reduced_cv <- pROC::auc(roc_reduced_cv)

cat("\nCross-Validated AUC vollständiges Modell:", as.numeric(auc_full_cv), "\n")
cat("Cross-Validated AUC reduziertes Modell:", as.numeric(auc_reduced_cv), "\n")

auc_table <- tibble(
  Modell = c("Vollständiges Modell", "Reduziertes Modell"),
  AUC = c(as.numeric(auc_full_cv), as.numeric(auc_reduced_cv))
)

write_csv(
  auc_table,
  "outputs/tables/cross_validated_auc.csv"
)

roc_to_dataframe <- function(roc_object, model_name) {
  data.frame(
    FPR = rev(1 - roc_object$specificities),
    TPR = rev(roc_object$sensitivities),
    Model = model_name
  )
}

roc_full_df <- roc_to_dataframe(roc_full_cv, "Vollständiges Modell")
roc_reduced_df <- roc_to_dataframe(roc_reduced_cv, "Reduziertes Modell")

roc_df <- bind_rows(roc_full_df, roc_reduced_df)

plot_roc <- ggplot(
  roc_df,
  aes(x = FPR, y = TPR, color = Model, linetype = Model)
) +
  geom_step(linewidth = 0.8) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dotted",
    color = "gray50"
  ) +
  scale_color_manual(
    values = c(
      "Vollständiges Modell" = "#2563EB",
      "Reduziertes Modell" = "#DC2626"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Vollständiges Modell" = "dashed",
      "Reduziertes Modell" = "solid"
    )
  ) +
  annotate(
    "text",
    x = 0.15,
    y = 0.22,
    hjust = 0,
    color = "#2563EB",
    size = 4,
    label = paste0(
      "Vollständiges Modell AUC: ",
      round(as.numeric(auc_full_cv), 3)
    )
  ) +
  annotate(
    "text",
    x = 0.15,
    y = 0.12,
    hjust = 0,
    color = "#DC2626",
    size = 4,
    label = paste0(
      "Reduziertes Modell AUC: ",
      round(as.numeric(auc_reduced_cv), 3)
    )
  ) +
  labs(
    x = "Falsch-Positiv-Rate (1 - Spezifität)",
    y = "Sensitivität (Wahre-Positiv-Rate)",
    color = "Modell",
    linetype = "Modell"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom"
  )

print(plot_roc)

ggsave(
  filename = "outputs/figures/07_cross_validated_roc.pdf",
  plot = plot_roc,
  width = 8,
  height = 6
)


# ------------------------------------------------------------
# 14. Kurze Ergebniszusammenfassung in der Konsole
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("ERGEBNISZUSAMMENFASSUNG\n")
cat("============================================================\n")

cat("\nZielvariable:\n")
cat("Harris = 1, Trump = 0\n")

cat("\nVollständiges Modell:\n")
cat("Formel:", deparse(full_formula), "\n")
cat("AIC:", round(AIC(full_model), 3), "\n")
cat("CV-AUC:", round(as.numeric(auc_full_cv), 3), "\n")

cat("\nReduziertes Modell:\n")
cat("Formel:", selection_result$best_formula, "\n")
cat("AIC:", round(AIC(reduced_model), 3), "\n")
cat("CV-AUC:", round(as.numeric(auc_reduced_cv), 3), "\n")

cat("\nSignifikante Variablen im reduzierten Modell auf 5%-Niveau:\n")
significant_variables <- reduced_model_coefficients |>
  filter(Variable != "(Intercept)", p_value < 0.05)

print(significant_variables)

cat("\nHinweis zur Interpretation:\n")
cat(
  "Die Ergebnisse beschreiben Assoziationen auf Bundesstaatenebene. ",
  "Sie erlauben keine kausalen Aussagen über individuelles Wahlverhalten.\n"
)

cat("\nAlle Tabellen und Grafiken wurden im Ordner outputs/ gespeichert.\n")