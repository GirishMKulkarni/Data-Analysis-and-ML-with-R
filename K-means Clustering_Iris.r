# Load necessary libraries
library(animation)

# K-means clustering
set.seed(123)
kc <- kmeans(traits, 3)
print(kc)

# Plot clusters and centers
plot(iris[c("Sepal.Length", "Sepal.Width")], col = colors[as.integer(species)], pch = kc$cluster)
points(kc$centers[, c("Sepal.Length", "Sepal.Width")], col = colors, pch = 1:3, cex = 3)
table(iris$Species, kc$cluster)

# Create and save a GIF animation of the clustering process
saveGIF(kmeans.ani(x = traits[, 1:2], col = colors), interval = 1, ani.width = 800, ani.height = 800)

# Function to get total within-cluster sum of squares for different cluster sizes
getSumSq <- function(k) { kmeans(traits, k, nstart = 25)$tot.withinss }

# Perform algorithm for different cluster sizes and retrieve variance
iris.kmeans1to10 <- sapply(1:10, getSumSq)
plot(1:10, iris.kmeans1to10, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares",
     col = c("black", "red", rep("black", 8)))
