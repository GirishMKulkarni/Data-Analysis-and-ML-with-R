library(keras)
library(tensorflow)

# Set random seed
set_random_seed(321L, disable_gpu = FALSE)  # Already sets R's random seed.

# Load flower dataset
data = EcoData::dataset_flower()

# Preprocess data
train = data$train/127.5 - 1 
test = data$test/127.5 - 1
labels = data$labels

# Load pretrained DenseNet201 model
pretrained_model = keras::application_densenet201(include_top = FALSE, input_shape = c(80L, 80L, 3L))

# Freeze weights of the pretrained model
keras::freeze_weights(pretrained_model)

# Build DNN model
dnn = pretrained_model$output %>% 
  layer_flatten() %>% 
  layer_dense(units = 5L, activation = "softmax")

# Create the final model
model = keras_model(inputs = pretrained_model$input, outputs = dnn)

# Set up data augmentation
aug = image_data_generator(rotation_range = 0, zoom_range = 0.4,
                           width_shift_range = 0.2, height_shift_range = 0.2,
                           horizontal_flip = TRUE)

# Set up the data
indices = sample.int(nrow(train), 0.1 * nrow(train)) # for validation
generator = flow_images_from_data(x = train[-indices,,,],
                                  y = k_one_hot(labels[-indices], 5L),
                                  generator = aug)

# Compile the model
model %>%
  compile(loss = loss_categorical_crossentropy,
          optimizer = keras::optimizer_sgd(learning_rate = 0.001))

# Set steps_per_epoch
steps_per_epoch = nrow(train[-indices,,,]) / 32
steps_per_epoch = floor(steps_per_epoch)

# Train the model
model %>% 
  fit(generator, epochs = 30L, batch_size = 32L, steps_per_epoch = steps_per_epoch, 
      validation_data = list(train[indices,,,], k_one_hot(labels[indices], 5L))
  )

# Make predictions on train set
predc = apply(model %>% predict(train), 1, which.max)
auc = Metrics::accuracy(predc - 1L, labels)
summary(predc)
auc

# Make predictions on test set
pred = apply(model %>% predict(test), 1, which.max)

# Save predictions to a CSV file
write.csv(data.frame(y = pred-1L), file = "densenet201_sgd.csv")
