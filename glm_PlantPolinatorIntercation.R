# Load necessary libraries
library(EcoData)
library(missRanger)
library(dplyr)
library(scales)  # Added library for scales
library(Metrics)  # Added library for Metrics

# Load dataset
data(plantPollinator_df)
plant_poll = plantPollinator_df
head(plant_poll)

# Impute missing values using missRanger
set.seed(123)
plant_poll_imputed = plant_poll %>%
  select(diameter, corolla, colour, nectar, tongue, body, interaction)

plant_poll_imputed = missRanger::missRanger(data = plant_poll_imputed %>%
                                              select(-interaction), verbose = 0)
plant_poll_imputed$interaction = plant_poll$interaction

# Data preprocessing
plant_poll_imputed =
  plant_poll_imputed %>%
  mutate(diameter = scales::rescale(diameter, c(0, 1))) %>%
  mutate(corolla = as.integer(as.factor(corolla)) - 1L,
         nectar = as.integer(as.factor(nectar)) - 1L,
         colour = as.integer(as.factor(colour)) - 1L) %>%
  mutate(tongue = ifelse(is.na(tongue) ,0, tongue),
         body = ifelse(is.na(body) ,0, body))

head(plant_poll_imputed)

# Split the data into training, validation, and test sets
train_data = plant_poll_imputed[!is.na(plant_poll_imputed$interaction), ]
indices = sample.int(nrow(train_data), 0.2 * nrow(train_data))  # for validation
train = train_data[-indices,]
valid = train_data[indices,]
test = plant_poll_imputed[is.na(plant_poll_imputed$interaction), ]

# Build a logistic regression model
model = glm(interaction ~ ., data = train, family = binomial())

# Make predictions on the validation set
preds = predict(model, newdata = valid, type = "response")
preds = ifelse(preds >= 0.5, 1, 0)

# Evaluate model accuracy on the validation set
auc = Metrics::accuracy(preds, valid$interaction)
auc

# Make predictions on the test set and save results to a CSV file
preds = predict(model, newdata = test, type = "response")
preds = ifelse(preds >= 0.5, 1, 0)
write.csv(data.frame(y = preds), file = "Girish_glm.csv")
