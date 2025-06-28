# -----------------------------------
# 1. Load Required Libraries
# -----------------------------------
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("factoextra")) install.packages("factoextra")

library(tidyverse)
library(ggplot2)
library(scales)
library(factoextra)

# -----------------------------------
# 2. Load and Clean the Data
# -----------------------------------
superstore <- read_csv("C:/Users/VINI/Downloads/archive (2)/Superstore.csv")

superstore <- superstore %>%
  rename_all(~make.names(.)) %>%
  mutate(
    Category = as.factor(Category),
    Sub.Category = as.factor(Sub.Category),
    Region = as.factor(Region)
  )

# -----------------------------------
# 3. Summary Statistics
# -----------------------------------
summary_stats <- superstore %>%
  summarise(
    Total_Sales = sum(Sales),
    Total_Profit = sum(Profit),
    Avg_Profit_Margin = mean(Profit / Sales, na.rm = TRUE)
  )
print(summary_stats)

# -----------------------------------
# 4. EDA Visualizations
# -----------------------------------

# 4.1 Sales by Region
ggplot(superstore, aes(x = Region, y = Sales, fill = Region)) +
  geom_bar(stat = "summary", fun = sum) +
  scale_y_continuous(labels = dollar) +
  labs(title = "Total Sales by Region", x = "Region", y = "Sales ($)")

# 4.2 Profit by Category
ggplot(superstore, aes(x = Category, y = Profit, fill = Category)) +
  geom_bar(stat = "summary", fun = sum) +
  scale_y_continuous(labels = dollar) +
  labs(title = "Total Profit by Category", x = "Category", y = "Profit ($)")

# 4.3 Profit by Sub-Category
ggplot(superstore, aes(x = reorder(Sub.Category, Profit), y = Profit, fill = Category)) +
  geom_bar(stat = "summary", fun = sum) +
  coord_flip() +
  scale_y_continuous(labels = dollar) +
  labs(title = "Profit by Sub-Category", x = "Sub-Category", y = "Profit ($)")

# 4.4 Sales vs Profit Scatter Plot
ggplot(superstore, aes(x = Sales, y = Profit, color = Category)) +
  geom_point(alpha = 0.6) +
  scale_x_continuous(labels = dollar) +
  scale_y_continuous(labels = dollar) +
  labs(title = "Sales vs Profit", x = "Sales ($)", y = "Profit ($)")

# -----------------------------------
# 5. Outlier Removal + Boxplots (Profit & Sales)
# -----------------------------------

# 5.1 Function to remove outliers by Category
remove_outliers_by_group <- function(df, group_col, value_col) {
  df %>%
    group_by(!!sym(group_col)) %>%
    filter({
      q1 <- quantile(.data[[value_col]], 0.25, na.rm = TRUE)
      q3 <- quantile(.data[[value_col]], 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      lower <- q1 - 1.5 * iqr
      upper <- q3 + 1.5 * iqr
      .data[[value_col]] >= lower & .data[[value_col]] <= upper
    }) %>%
    ungroup()
}

# 5.2 Apply to Profit and Sales
filtered_profit <- remove_outliers_by_group(superstore, "Category", "Profit")
filtered_sales <- remove_outliers_by_group(superstore, "Category", "Sales")

# 5.3 Boxplot of Profit (outliers removed)
ggplot(filtered_profit, aes(x = Category, y = Profit, fill = Category)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(labels = dollar) +
  labs(title = "Boxplot of Profit by Category (Outliers Removed)", x = "Category", y = "Profit ($)") +
  theme_minimal()

# 5.4 Boxplot of Sales (outliers removed)
ggplot(filtered_sales, aes(x = Category, y = Sales, fill = Category)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(labels = dollar) +
  labs(title = "Boxplot of Sales by Category (Outliers Removed)", x = "Category", y = "Sales ($)") +
  theme_minimal()

# -----------------------------------
# 6. Linear Regression
# -----------------------------------
lm_model <- lm(Profit ~ Sales + Discount, data = superstore)
summary(lm_model)

# 6.1 Profit vs Discount
ggplot(superstore, aes(x = Discount, y = Profit)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Linear Regression: Profit vs Discount", x = "Discount", y = "Profit ($)")

# 6.2 Profit vs Sales
ggplot(superstore, aes(x = Sales, y = Profit)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Linear Regression: Profit vs Sales", x = "Sales ($)", y = "Profit ($)")

# -----------------------------------
# 7. K-Means Clustering
# -----------------------------------

# 7.1 Select and Scale Data
cluster_data <- superstore %>%
  select(Sales, Profit, Discount) %>%
  drop_na() %>%
  filter(Sales < quantile(Sales, 0.99), Profit > quantile(Profit, 0.01))

cluster_scaled <- scale(cluster_data)

# 7.2 Apply K-means
set.seed(123)
kmeans_result <- kmeans(cluster_scaled, centers = 3, nstart = 25)
cluster_data$Cluster <- as.factor(kmeans_result$cluster)

# 7.3 Visualize Clusters
fviz_cluster(kmeans_result, data = cluster_scaled,
             geom = "point",
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             main = "K-means Clustering: Sales, Profit, Discount")
