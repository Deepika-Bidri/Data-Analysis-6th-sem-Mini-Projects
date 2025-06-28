# Superstore Sales Analysis Using R

## 📌 Project Title:
**Superstore Sales Analysis Using R**

## 🏫 Institution:
**P.E.S. College of Engineering, Mandya – 571401**  
(An Autonomous Institution under VTU, Belgaum)

## 👩‍🏫 Under the Guidance of:
**Dr. Deepika Bidri**  
Asst. Professor, Dept. of CS&E  
P.E.S.C.E, Mandya

## 👨‍💻 Submitted by:
- **Mahadevaswamy M R** [USN: 4PS22CS198]  
- **Likith D** [USN: 4PS22CS199]  
- **Hemanth M U** [USN: 4PS22CS201]  
- **Praveen Kumar R** [USN: 4PS23CS412]  
- **Yashwanth Gowda N R** [USN: 4PS23CS415]

---

## 📖 Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Tools and Technologies](#tools-and-technologies)
5. [Project Features](#project-features)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

---

## 📘 Introduction

### Overview of the Business Problem
In the modern retail industry, large volumes of transactional data are generated through customer purchases, sales, discounts, and regional performance. However, much of this data is not utilized to its full potential. This project focuses on turning this raw data into meaningful business insights using data analysis and machine learning techniques in R.

### Importance of Data Analysis in Retail
Data analytics allows businesses to:
- Track sales, profit, and performance indicators.
- Understand the effect of discount strategies.
- Identify best and worst performing product categories or regions.
- Perform customer segmentation to optimize marketing and sales strategies.

---

## 🎯 Objectives

1. **Sales & Profit Analysis:**  
   Analyze how sales and profit vary across regions and categories.

2. **Discount Impact:**  
   Investigate the relationship between discounts and profit using regression.

3. **Predictive Modeling:**  
   Build a linear regression model to predict profit based on sales and discount.

4. **Customer Segmentation:**  
   Apply K-means clustering to group similar transactions.

5. **Data Visualization:**  
   Use `ggplot2` and `factoextra` to visualize key insights and trends.

---

## 📊 Dataset

- **Name:** Sample - Superstore.csv  
- **Source:** Tableau (default dataset) / Kaggle  
- **Features:**
  - Sales, Profit, and Discount
  - Region, Category, Sub-Category
  - Ship Date, Order Date, Customer ID, Segment, etc.

---

## 🧰 Tools and Technologies

### Language:
- **R** – for statistical computing and data visualization.

### IDE:
- **RStudio** – for scripting and analysis.

### Libraries Used:
- `tidyverse` – for data wrangling and visualization.
  - `dplyr`, `ggplot2`, `readr`, `tibble`, `stringr`, `purrr`
- `scales` – for formatting values (currency, %).
- `factoextra` – for visualizing clustering analysis (K-means).

---

## ✨ Project Features

- Bar plots for Sales and Profit by Region, Category, Sub-Category.
- Box plots to show discount impact on profit.
- Correlation heatmaps and scatter plots.
- Linear regression model with statistical interpretation.
- K-Means clustering for customer segmentation.
- Clean, interactive visualizations using `ggplot2`.

---

## 🛠 Installation & Setup

1. **Install R and RStudio:**
   - Download from [https://cran.r-project.org](https://cran.r-project.org)
   - RStudio from [https://posit.co/download/rstudio-desktop](https://posit.co/download/rstudio-desktop)

2. **Clone the Project or Download ZIP:**
   ```bash
   git clone https://github.com/your-username/superstore-sales-analysis.git

