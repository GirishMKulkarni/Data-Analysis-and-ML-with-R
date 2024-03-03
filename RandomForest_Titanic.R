# Random forest

# Load required libraries
library(EcoData)
library(dplyr)
library(missRanger)
library(Matrix)
library(glmnet)
library(tree)
library(randomForest)

# Load Titanic dataset
data(titanic_ml)
data = titanic_ml

# Select relevant columns
data = data %>% select(survived, sex, age, fare, pclass)

# Impute missing values using missRanger
data[,-1] = missRanger(data[,-1], verbose = 0)

# Feature scaling and transformation
data_sub =
  data %>%
  mutate(age = scales::rescale(age, c(0, 1)),
         fare = scales::rescale(fare, c(0, 1))) %>%
  mutate(sex = as.integer(sex) - 1L,
         pclass = as.integer(pclass - 1L))

# Create new datasets for prediction and observation
data_new = data_sub[is.na(data_sub$survived),]
data_obs = data_sub[!is.na(data_sub$survived),]

# Hyperparameter optimization

# Set seed for reproducibility
set.seed(42)

# Define number of cross-validation folds
cv = 3

# Create indices for cross-validation
outer_split = as.integer(cut(1:nrow(data_obs), breaks = cv))

# Sample minnodesize and mtry values
hyper_minnodesize = sample(120, 20)
hyper_mtry = sample(5, 20, replace = TRUE)

# Initialize results dataframe
results = data.frame(
  set = rep(NA, cv),
  minnodesize = rep(NA, cv),
  mtry = rep(NA, cv),
  AUC = rep(NA, cv)
)

# Cross-validation loop
for(i in 1:cv) {
  train_outer = data_obs[outer_split != i, ]
  test_outer = data_obs[outer_split == i, ]

  tuning_results =
    sapply(1:length(hyper_minnodesize), function(k) {
      model = randomForest(as.factor(survived)~., data = train_outer, mtry = hyper_mtry[k], nodesize = hyper_minnodesize[k])
      return(Metrics::auc(test_outer$survived, predict(model, newdata = test_outer, type = "prob")[,2]))
    })
  best_minnodesize = hyper_minnodesize[which.max(tuning_results)]
  best_mtry = hyper_mtry[which.max(tuning_results)]

  results[i, 1] = i
  results[i, 2] = best_minnodesize
  results[i, 3] = best_mtry
  results[i, 4] = max(tuning_results)
}

# Display optimization results
print(results)

# Predictions & submissions

# Train the final model on the entire observed dataset
model = randomForest(as.factor(survived)~., data = data_obs, nodesize = 26, mtry = 2)

# Extract data for predictions
data_new = data_sub[is.na(data_sub$survived),]

# Generate predictions and save to a CSV file
write.csv(data.frame(y = predict(model, data_new, type = "prob")[,2]), file = "Group8_rf_best_model.csv")

# Result

# Create an ensemble of models and generate predictions
prediction_ensemble =
  sapply(1:nrow(results), function(i) {
    model = randomForest(as.factor(survived)~., data = data_obs, nodesize = results$minnodesize[i], mtry = results$mtry[i])
    return(predict(model, data_obs, nodesize = results$minnodesize[i], mtry = results$mtry[i], type = "prob")[,2])
  })

# Calculate the mean of ensemble predictions
write.csv(data.frame(y = apply(prediction_ensemble, 1, mean)), file = "Group8_rf_ensemble.csv")
