**### Heart Rate Analysis and Heart Disease Risk Prediction**

**Overview**
This repository contains the analysis and findings of a heart disease study using the Cleveland Heart Disease dataset. The primary goal of this project is to investigate how maximum heart rate during exercise (thalach) is associated with the likelihood of having heart disease, while controlling for other clinical variables such as age and sex.

Through a blend of exploratory data analysis, statistical testing, and logistic regression modeling, this project identifies key risk factors and provides actionable insights for healthcare screening and predictive modeling in clinical settings.

**Project Structure**

The main output of this project includes:

A logistic regression model evaluating key risk factors.

Statistical test outputs for each clinical variable.

Visualizations demonstrating variable relationships.

Model performance metrics (Accuracy, AUC, Confusion Matrix).

A prediction tool that estimates the probability of heart disease based on individual input.

Dataset
Source: Cleveland Heart Disease dataset
File Used: Cleveland_hd.csv
Number of Records: 303 observations
Key Variables:

age: Age of the patient

sex: Gender (0 = Female, 1 = Male)

thalach: Maximum heart rate achieved during exercise

class: Heart disease diagnosis (0 = No disease, 1-4 = Presence of disease)

Additional variables include cholesterol, blood pressure, chest pain type, etc.

**Methodology**

The analysis consists of the following key steps:

**🔹 Data Cleaning & Transformation**

Converted the class variable into a binary outcome (hd).

Re-coded sex as a factor variable with labels "Male" and "Female".

Removed or handled missing/invalid values.

🔹 **Statistical Analysis**

Chi-Squared Test: To assess the relationship between sex and heart disease.

T-Tests: For continuous variables like age and thalach to compare means between disease groups.

Boxplots and Barplots: To visualize group differences across variables.

**🔹 Logistic Regression Modeling**
A multivariate logistic regression model was constructed using:


glm(hd ~ age + sex + thalach, family = "binomial")
This model estimates the odds of having heart disease based on the three predictors.

**🔹 Odds Ratio Interpretation**

Odds Ratios (OR) and 95% Confidence Intervals (CI) were computed to quantify how each predictor influences the likelihood of heart disease.

**🔹 Prediction**

Used the model to predict heart disease probability for a new case (e.g., 45-year-old female with thalach = 150).

**🔹 Model Performance Metrics**

AUC: 0.706

Accuracy: ~71%

Classification Error: ~29%

Confusion Matrix: 2x2 matrix comparing true and predicted labels.

**Key Findings**

Maximum Heart Rate (thalach) is negatively associated with heart disease risk.

Older age and being male significantly increase the odds of heart disease.

The final logistic model demonstrates fair predictive performance with practical clinical utility.

For example, a 45-year-old female with a max heart rate of 150 has a predicted disease probability of 17.7%, indicating low risk.

**Conclusion**

This project demonstrates how statistical modeling can aid in early detection of heart disease using interpretable clinical features. The model and methods can be further extended with additional predictors or used as a decision support tool in healthcare settings.

**Submitted by:**

VILAS GOWDA T [USN:4PS22CS190] 

VARSHINI [USN:4PS22CS197] 

NELSON [USN:4PS23CS411] 

ROHAN M K [USN:4PS23CS413] 

THASHWINI R [USN:4PS23CS414] 

**Under the Guidance of:**

Prof. Deepika B

Assistant Professor, Dept. of CS&E,

P.E.S.C.E, Mandya.

P.E.S. College of Engineering, Mandya

Department of Computer Science and Engineering

2024-2025
