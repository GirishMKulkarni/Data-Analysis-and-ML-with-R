# XGBoost

# Load required libraries
library(EcoData)
library(dplyr)
library(missRanger)
library(Matrix)
library(xgboost)

# Load Titanic dataset
data(titanic_ml)
data = titanic_ml

# Select relevant columns
data %>% select(survived, sex, age, fare, pclass, sibsp, parch, boat)
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
data_obs = data_sub[!is.na(data_sub$survived),]

# Hyperparameter optimization for XGBoost

# Set seed for reproducibility
set.seed(42)
data_clean = data_sub[!is.na(data_sub$survived),]
indices = sample.int(nrow(data_clean), 0.1 * nrow(data_clean)) # for validation
data_train =data_clean[-indices,,]
data_test = data_clean[indices,,]
dim(data_train)
dim(data_test)
dim(data_clean)
length(indices)



hypers=expand.grid(max_depth = c(5),
                   eta = c(0.03),
                   subsample = c(0.7),
                   lambda = c(1))
dim(hypers)

best_hypers = data.frame(
  nrounds = 0 ,
  max_depth = 0,
  eta =0,
  subsample = 0,
  lambda = 0,
  test_rmse = 0 
)

# Define number of cross-validation folds
cv = 30

# Create indices for cross-validation
outer_split = as.integer(cut(1:nrow(data_obs), breaks = cv))

# Sample hyperparameter values
hyper_max_depth = sample(5:15, 20, replace = TRUE)
hyper_eta = runif(20, 0.01, 0.3)
hyper_colsample_bytree = runif(20, 0.5, 1)

# Initialize results dataframe
results_xgboost = data.frame(
  set = rep(NA, cv),
  max_depth = rep(NA, cv),
  eta = rep(NA, cv),
  cols = rep(NA, cv),
  AUC = rep(NA, cv)
)

# Cross-validation loop for XGBoost
tuning_results =
  sapply(1:dim(hypers)[1], function(k) {
    brt_cv = xgboost::xgb.cv(data=xgb.DMatrix(data = as.matrix(scale(data_train[,-1])), label = data_train$survived),
                             nfold = 3L,
                             nrounds = 500L, 
                             max_depth=hypers$max_depth[k],
                             eta = hypers$eta[k],
                             subsample =hypers$subsample[k],
                             lambda =hypers$lambda[k],
                             #objective="binary:logistic"
    )
    return(cbind(which.min(brt_cv$evaluation_log$test_rmse_mean),
                 brt_cv$evaluation_log$test_rmse_mean[which.min(brt_cv$evaluation_log$test_rmse_mean)]))
  })

least_test_rmse = which.min(tuning_results[2, ])
best_hypers[1, 1] = tuning_results[1,least_test_rmse]
best_hypers[1, 2] = hypers$max_depth[least_test_rmse]
best_hypers[1, 3] = hypers$eta[least_test_rmse]
best_hypers[1, 4] = hypers$subsample[least_test_rmse]
best_hypers[1, 5] = hypers$lambda[least_test_rmse]
best_hypers[1, 6] = tuning_results[2,least_test_rmse]
#results[1, 4] = max(tuning_results)
# Display XGBoost optimization results
print(results_xgboost)

# Train the final XGBoost model
final_model_xgboost = xgboost(data = as.matrix(data_obs[,-1]), label = data_obs$survived,
                              max_depth = 10, eta = 0.1, colsample_bytree = 0.8)

# Extract data for XGBoost predictions
data_new_xgboost = as.matrix(data_sub[is.na(data_sub$survived),-1])

# Generate XGBoost predictions and save to a CSV file
write.csv(data.frame(y = predict(final_model_xgboost, data_new_xgboost)), file = "Group8_xgboost_best_model.csv")

