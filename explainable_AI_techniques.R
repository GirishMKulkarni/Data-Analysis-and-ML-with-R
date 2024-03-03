# Explorable Elephant Occurrence Data with XGBoost and Interpretable ML

# Load necessary libraries
library(iml)
library(xgboost) 
library(EcoData)
library(partykit)  # Required for the global explainer
library(yaImpute)

# Set seed for reproducibility
set.seed(123)

# Load elephant occurrence data
data <- EcoData::elephant$occurenceData
head(data)

# Explore the dataset
?EcoData::elephant

# Bioclim variables explanation
# | Variable | Meaning                                              |
# |----------|------------------------------------------------------|
# | bio1     | Annual Mean Temperature                              |
# | bio2     | Mean Diurnal Range (Mean of monthly max-min temp)    |
# | ...      | ...                                                  |
# | bio19    | Precipitation of Coldest Quarter                     |

# Create XGBoost data object
data_xg <- xgb.DMatrix(
  data = as.matrix(data[,-1]),
  label = data$Presence
)

# Train XGBoost model
brt <- xgboost(data_xg, nrounds = 50, objective = "binary:logistic")

## Feature importance
# Built-in feature importance
xgboost::xgb.importance(model = brt)

# Create a predict wrapper function for interpretability
predict_wrapper <- function(model, newdata) {
  return(predict(model, as.matrix(newdata)))
}

# Test the predict wrapper
predict_wrapper(brt, data[,-1])

# Create a Predictor object for interpretability
predictor <- Predictor$new(
  brt,
  data = data[,-1],
  y = data[,1],
  predict.function = predict_wrapper
)

# Variable importance
imp <- FeatureImp$new(predictor, loss = "logLoss")
plot(imp)

# Partial dependency plots
eff <- FeatureEffect$new(predictor, feature = "bio16", method = "pdp")
plot(eff)

eff <- FeatureEffect$new(predictor, feature = "bio16", method = "pdp+ice")
plot(eff)

# ALE plots
ale <- FeatureEffect$new(predictor, feature = "bio16", method = "ale")
plot(ale)

# Feature interactions
interact <- Interaction$new(predictor, feature = "bio16")
plot(interact)

ale <- FeatureEffect$new(predictor, feature = c("bio16", "bio3"), method = "ale")
plot(ale)

## Global explainer
# Make the model interpretable with a surrogate decision tree
tree <- TreeSurrogate$new(predictor, maxdepth = 3L)
plot(tree$tree)

## Local explainer
# SHAP (Shapley Additive exPlanations) values for local interpretability
shapley <- Shapley$new(predictor, x.interest = data[100,-1])
plot(shapley)
