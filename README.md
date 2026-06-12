# US Election 2024 – Logistic Regression Analysis / US-Wahl 2024 – Logistische Regressionsanalyse

## English Version

This project analyzes the 2024 US presidential election at the state level using demographic and socioeconomic variables.

The goal is to examine which factors distinguish states won by Harris from states won by Trump.

## Project Overview

The analysis uses state-level data and applies logistic regression to classify whether a state was won by Harris or Trump.

The target variable is binary:

```text
Harris = 1
Trump = 0
```

The project combines exploratory data analysis, statistical modeling, feature selection and model evaluation.

## Research Question

Which demographic and socioeconomic state-level factors are associated with whether a state was won by Harris or Trump in the 2024 US presidential election?

## Dataset

The dataset contains state-level information such as:

* Population
* Total area
* Population density
* Median age
* Birth rate
* Human Development Index
* Unemployment rate
* Health insurance coverage
* Median rent
* Election winner

The analysis is based on aggregated state-level data. Therefore, the results should not be interpreted as individual voting behavior.

## Methods

The following methods were used:

* Data cleaning and preprocessing
* Exploratory data analysis
* Logistic regression
* AIC-based model selection
* Odds ratio interpretation
* Confidence intervals
* Cross-validation
* ROC/AUC evaluation

## Models

Two logistic regression models were compared.

### Full Model

The full model included all considered predictors:

```text
log_total_area
log_population
median_age
birth_rate
hdi
unemployment_rate_pct
health_insurance_coverage_pct
median_rent
```

### Reduced Model

A reduced model was selected using AIC-based model selection.

The reduced model included:

```text
log_population
median_age
unemployment_rate_pct
health_insurance_coverage_pct
median_rent
```

The reduced model achieved a lower AIC and a higher cross-validated AUC than the full model.

## Key Results

In the reduced logistic regression model, the following variables were statistically significant at the 5% level:

* Health insurance coverage
* Median rent

Both variables showed a positive association with Harris states.

This means that, in the reduced model, states with higher health insurance coverage and higher median rent were more likely to be classified as Harris states.

## Important Interpretation Note

The results describe statistical associations at the state level.

They do not imply causality and do not allow conclusions about individual voting behavior.

This is an exploratory analysis based on aggregated data.

## Technologies Used

* Python
* R
* pandas
* statsmodels
* scikit-learn
* Logistic regression
* ROC/AUC
* Cross-validation
* Jupyter Notebook

## Project Files

```text
US_election_2024.csv
US_election_2024_cleaned.csv
us_election_2024.py
us_election_2024.R
us_election_2024_analysis.ipynb
outputs/
README.md
```

## How to Run

Install the required Python packages:

```bash
pip install pandas statsmodels scikit-learn matplotlib seaborn
```

Run the Python script:

```bash
python us_election_2024.py
```

Or open the Jupyter Notebook:

```text
us_election_2024_analysis.ipynb
```

The R script can be opened and executed in RStudio:

```text
us_election_2024.R
```

## Conclusion

The reduced logistic regression model performed better than the full model based on AIC and cross-validated AUC.

The strongest statistical associations in the reduced model were found for health insurance coverage and median rent. These variables helped distinguish Harris states from Trump states in the analyzed state-level dataset.

The analysis should be interpreted as a descriptive and exploratory machine learning/statistical modeling project, not as a causal explanation of the election outcome.

---

# Deutsche Version

Dieses Projekt analysiert die US-Präsidentschaftswahl 2024 auf Bundesstaatenebene anhand demografischer und sozioökonomischer Variablen.

Ziel ist es zu untersuchen, welche Faktoren Bundesstaaten unterscheiden, die von Harris beziehungsweise Trump gewonnen wurden.

## Projektüberblick

Die Analyse basiert auf Daten auf Bundesstaatenebene und verwendet logistische Regression, um zu klassifizieren, ob ein Bundesstaat von Harris oder Trump gewonnen wurde.

Die Zielvariable ist binär kodiert:

```text
Harris = 1
Trump = 0
```

Das Projekt kombiniert explorative Datenanalyse, statistische Modellierung, Variablenauswahl und Modellevaluation.

## Forschungsfrage

Welche demografischen und sozioökonomischen Faktoren auf Bundesstaatenebene hängen damit zusammen, ob ein Bundesstaat bei der US-Präsidentschaftswahl 2024 von Harris oder Trump gewonnen wurde?

## Datensatz

Der Datensatz enthält Informationen auf Bundesstaatenebene, darunter:

* Bevölkerung
* Gesamtfläche
* Bevölkerungsdichte
* Medianes Alter
* Geburtenrate
* Human Development Index
* Arbeitslosenrate
* Krankenversicherungsquote
* Mediane Miete
* Wahlsieger

Die Analyse basiert auf aggregierten Daten auf Bundesstaatenebene. Daher erlauben die Ergebnisse keine direkten Aussagen über individuelles Wahlverhalten.

## Methoden

Folgende Methoden wurden verwendet:

* Datenbereinigung und Preprocessing
* Explorative Datenanalyse
* Logistische Regression
* AIC-basierte Modellauswahl
* Interpretation von Odds Ratios
* Konfidenzintervalle
* Cross-Validation
* ROC/AUC-Evaluation

## Modelle

Es wurden zwei logistische Regressionsmodelle verglichen.

### Vollständiges Modell

Das vollständige Modell enthielt alle betrachteten Prädiktoren:

```text
log_total_area
log_population
median_age
birth_rate
hdi
unemployment_rate_pct
health_insurance_coverage_pct
median_rent
```

### Reduziertes Modell

Das reduzierte Modell wurde mithilfe einer AIC-basierten Modellauswahl bestimmt.

Das reduzierte Modell enthielt:

```text
log_population
median_age
unemployment_rate_pct
health_insurance_coverage_pct
median_rent
```

Das reduzierte Modell erzielte einen niedrigeren AIC-Wert und eine höhere kreuzvalidierte AUC als das vollständige Modell.

## Zentrale Ergebnisse

Im reduzierten logistischen Regressionsmodell waren folgende Variablen auf dem 5%-Niveau statistisch signifikant:

* Krankenversicherungsquote
* Mediane Miete

Beide Variablen zeigten einen positiven Zusammenhang mit Harris-Staaten.

Das bedeutet: Im reduzierten Modell wurden Bundesstaaten mit höherer Krankenversicherungsquote und höherer medianer Miete eher als Harris-Staaten klassifiziert.

## Wichtiger Interpretationshinweis

Die Ergebnisse beschreiben statistische Zusammenhänge auf Bundesstaatenebene.

Sie zeigen keine Kausalität und erlauben keine Aussagen über individuelles Wahlverhalten.

Die Analyse ist daher als exploratives Machine-Learning- und Statistikprojekt zu verstehen.

## Verwendete Technologien

* Python
* R
* pandas
* statsmodels
* scikit-learn
* Logistische Regression
* ROC/AUC
* Cross-Validation
* Jupyter Notebook

## Projektdateien

```text
US_election_2024.csv
US_election_2024_cleaned.csv
us_election_2024.py
us_election_2024.R
us_election_2024_analysis.ipynb
outputs/
README.md
```

## Ausführung

Benötigte Python-Pakete installieren:

```bash
pip install pandas statsmodels scikit-learn matplotlib seaborn
```

Python-Skript ausführen:

```bash
python us_election_2024.py
```

Oder das Jupyter Notebook öffnen:

```text
us_election_2024_analysis.ipynb
```

Das R-Skript kann in RStudio geöffnet und ausgeführt werden:

```text
us_election_2024.R
```

## Fazit

Das reduzierte logistische Regressionsmodell schnitt anhand von AIC und kreuzvalidierter AUC besser ab als das vollständige Modell.

Die stärksten statistischen Zusammenhänge im reduzierten Modell zeigten sich für die Krankenversicherungsquote und die mediane Miete. Diese Variablen halfen dabei, Harris- und Trump-Staaten im analysierten Datensatz zu unterscheiden.

Die Ergebnisse sollten als deskriptive und explorative Analyse interpretiert werden, nicht als kausale Erklärung des Wahlausgangs.
