# AI-Powered Job Market Data Analysis and Prediction

## 📌 Overview
This project explores how Artificial Intelligence (AI) is transforming the global job market. By analyzing job-related data using data science and machine learning techniques, we aim to uncover salary trends, work patterns (remote, hybrid, on-site), and the automation risk for various job roles. The project provides insights that can guide job seekers, employers, and policymakers in understanding AI's impact on employment.

## 🎯 Objectives
- Analyze how AI adoption and automation risk influence job roles and salaries.
- Identify patterns based on job categories, locations (remote/on-site), and regions.
- Predict automation risk levels using machine learning models.
- Recommend future-ready job roles and high-potential regions.

## 📊 Dataset
- **Source**: Kaggle (synthetic dataset reflecting AI job market trends).
- **Key Features**:
  - Job_Category, Work_Location, Country, City, Continent
  - Salary_USD
  - AI_Adoption_Level (1–3 scale)
  - Automation_Risk (categorical: Low, Medium, High)
  - AI_Impact_Score (engineered feature)

## 🧠 Machine Learning Model
- **Model Used**: Random Forest Classifier
- **Target Variable**: `Automation_Risk`
- **Key Steps**:
  - Data cleaning (handling nulls, encoding categorical variables)
  - Feature engineering
  - Splitting dataset into training and testing sets
  - Model evaluation using accuracy, confusion matrix, and classification report
  - Feature importance analysis to highlight critical predictors

## 📈 Data Analysis & Visualization
- Tools like histograms, heatmaps, and boxplots used to analyze:
  - Salary distributions
  - Regional salary differences
  - Remote vs on-site job compensation
  - Role-specific automation risks

## 🧰 Tools & Technologies
- **Language**: Python
- **IDE**: Jupyter Notebook
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation and numerical computation
  - `matplotlib`, `seaborn`: Data visualization
  - `scikit-learn`: Machine learning and model evaluation
- **Platform**: Kaggle (dataset source & benchmarking)

## 👥 Team Members
- **Sudarshan K** [USN: 4PS22CS162]  
- **Sushmitha H Y** [USN: 4PS22CS168]  
- **Tanushree S** [USN: 4PS22CS173]  
- **Varshanth Gowda M L** [USN: 4PS22CS185]  
- **Vinay C N** [USN: 4PS22CS192]

## 🧑‍🏫 Guide
**Dr. Deepika Bidri**  
Assistant Professor, Department of CSE  
P.E.S. College of Engineering, Mandya

## 📅 Academic Year
**2024–2025**

---
