library(dplyr)
library(ggplot2)
library(reshape2)


# Load the iris dataset
data(iris)

# Display the structure of the dataset
str(iris)

# Summary statistics
summary(iris)

# Head of the dataset
head(iris)

# Descriptive statistics by group (in this case, by species)
grouped_summary <- iris %>%
  group_by(Species) %>%
  summarise(
    mean_sepal_length = mean(Sepal.Length),
    sd_sepal_length = sd(Sepal.Length),
    mean_petal_length = mean(Petal.Length),
    sd_petal_length = sd(Petal.Length)
  )

print(grouped_summary)

# Visualization: Scatter plot of Sepal.Length vs. Sepal.Width colored by Species
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(title = "Scatter Plot of Sepal Length vs. Sepal Width by Species")

# Visualization: Boxplot of Petal.Length by Species
ggplot(iris, aes(x = Species, y = Petal.Length, fill = Species)) +
  geom_boxplot() +
  labs(title = "Boxplot of Petal Length by Species")

# Correlation matrix
cor_matrix <- cor(iris[, 1:4])
print(cor_matrix)
melted_cormat <- melt(cor_matrix)
# Visualization: Correlation heatmap
ggplot(data = as.data.frame(melted_cormat), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  theme_minimal() +
  labs(title = "Correlation Heatmap")

# Histogram of Sepal.Length
ggplot(iris, aes(x = Sepal.Length)) +
  geom_histogram(binwidth = 0.2, fill = "lightblue", color = "black") +
  labs(title = "Histogram of Sepal Length", x = "Sepal Length")

# Pair plot
pairs(iris[, 1:4], col = iris$Species, pch = 19)




# Load necessary libraries
library(ggplot2)
library(GGally)

# Load data
dat <- airquality

# Display structure of data
str(dat)

# View the first few rows of the data
head(dat)

# Summary statistics
summary(dat)

# Scatter plot of Wind vs. Temp for the first 100 rows
trunc_dat <- dat[1:100, ]
trunc_dat <- na.omit(trunc_dat)  # Remove NA values

# Scatter plots
plot(Wind ~ Temp, trunc_dat)
plot(Month ~ Temp, trunc_dat)

# Pairwise plots
pairs(dat)

# Pearson correlation
cor(dat$Wind, dat$Temp, use = "complete.obs")

# Spearman correlation
cor(dat$Wind, dat$Temp, use = "complete.obs", method = "spearman")

# Create a ggplot pairs matrix
pm <- ggpairs(dat)
print(pm)

# Load necessary libraries
library(ggplot2)

# Load sample data
data(flea)
head(flea)
str(flea)

# Scatterplot matrix with ggplot2
ggpairs(flea)

# Specify color by species in scatterplot matrix
ggpairs(flea, columns = 2:4, ggplot2::aes(colour = species))

# Load tips dataset
data(tips, package = "reshape")
head(tips)

# Scatterplot matrix with density plots
ggpairs(tips, upper = list(continuous = "density", combo = "box_no_facet"))
