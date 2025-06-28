# Load necessary libraries
library(tidyverse)
library(broom)
library(Metrics)

# 1. Load the dataset
Heartrate <- read.csv("E:/Downloads/da project/da project/Heartrate.csv", stringsAsFactors = FALSE)

# 2. Basic cleaning - replace "?" with NA and remove rows with NA
Heartrate[Heartrate == "?"] <- NA
Heartrate <- Heartrate %>%
  mutate(
    age = as.numeric(age),
    thalach = as.numeric(thalach),
    gender = as.numeric(gender),
    class = as.numeric(class)
  )
Heartrate <- na.omit(Heartrate)

# 3. Remove duplicate rows
Heartrate <- Heartrate %>% distinct()

# 4. Outlier detection and removal using IQR method for numeric variables
remove_outliers <- function(df, column) {
  Q1 <- quantile(df[[column]], 0.25)
  Q3 <- quantile(df[[column]], 0.75)
  IQR <- Q3 - Q1
  df %>% filter(df[[column]] >= (Q1 - 1.5 * IQR) & df[[column]] <= (Q3 + 1.5 * IQR))
}

Heartrate <- remove_outliers(Heartrate, "age")
Heartrate <- remove_outliers(Heartrate, "thalach")

# 5. Recode target variable and gender
Heartrate <- Heartrate %>%
  mutate(
    hd = ifelse(class > 0, 1, 0),
    gender = factor(gender, levels = c(0, 1), labels = c("Female", "Male")),
    hd_labelled = factor(ifelse(hd == 1, "Disease", "No Disease"))
  )

# 6. Statistical tests
hd_gender <- chisq.test(table(Heartrate$hd, Heartrate$gender))
hd_age <- t.test(age ~ hd, data = Heartrate)
hd_thalach <- t.test(thalach ~ hd, data = Heartrate)

print(hd_gender)
print(hd_age)
print(hd_thalach)

# 7. Visualizations
ggplot(Heartrate, aes(x = hd_labelled, y = age)) +
  geom_boxplot() +
  ggtitle("Age vs Heart Disease")

ggplot(Heartrate, aes(x = hd_labelled, fill = gender)) +
  geom_bar(position = "fill") +
  ylab("Proportion") +
  ggtitle("Gender vs Heart Disease")

ggplot(Heartrate, aes(x = hd_labelled, y = thalach)) +
  geom_boxplot() +
  ggtitle("Max Heart Rate vs Heart Disease")

# 8. Logistic regression model
model <- glm(hd ~ age + gender + thalach, data = Heartrate, family = "binomial")
summary(model)

# 9. Odds ratios with 95% confidence intervals
tidy_model <- tidy(model) %>%
  mutate(
    OR = exp(estimate),
    lower_CI = exp(estimate - 1.96 * std.error),
    upper_CI = exp(estimate + 1.96 * std.error)
  )
print(tidy_model)

# 10. Predict probabilities and decisions
Heartrate$pred_prob <- predict(model, newdata = Heartrate, type = "response")
Heartrate$pred_hd <- ifelse(Heartrate$pred_prob >= 0.5, 1, 0)

# 11. Prediction for a new individual
new_patient <- data.frame(age = 45, gender = "Female", thalach = 150)
pred_new <- predict(model, new_patient, type = "response")
print(paste("Predicted probability for new patient:", round(pred_new, 3)))

# 12. Model evaluation
model_auc <- auc(Heartrate$hd, Heartrate$pred_prob)
model_accuracy <- accuracy(Heartrate$hd, Heartrate$pred_hd)
model_ce <- ce(Heartrate$hd, Heartrate$pred_hd)

print(paste("AUC:", round(model_auc, 3)))
print(paste("Accuracy:", round(model_accuracy, 3)))
print(paste("Classification Error:", round(model_ce, 3)))

# 13. Confusion matrix
conf_matrix <- table(True = Heartrate$hd, Predicted = Heartrate$pred_hd)
print(conf_matrix)
