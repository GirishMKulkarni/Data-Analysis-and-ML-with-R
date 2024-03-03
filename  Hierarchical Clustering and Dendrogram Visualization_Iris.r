# Load necessary libraries
library(ape)

# Load iris dataset
set.seed(123)
iris <- datasets::iris
colors <- hcl.colors(3)
traits <- as.matrix(iris[, 1:4])
species <- iris$Species

# Hierarchical clustering and dendrogram
d <- dist(traits)
hc <- hclust(d, method = "complete")

# Plot dendrogram
plot(hc, main = "", group = iris$Species)
rect.hclust(hc, k = 3)  # Draw rectangles around the branches

# Plot dendrogram as phylogenetic tree
plot(as.phylo(hc), tip.color = colors[as.integer(species)], direction = "downwards")
rect.hclust(hc, k = 3)  # Draw rectangles around the branches

# Scaled hierarchical clustering
traits_scaled <- scale(traits)
d_scaled <- dist(traits_scaled)
hc_scaled <- hclust(d_scaled, method = "ward.D2")

# Plot scaled dendrogram
plot(as.phylo(hc_scaled), tip.color = colors[as.integer(species)], direction = "downwards")
rect.hclust(hc_scaled, k = 3)

# Cut dendrogram tree into groups
hcRes3 <- cutree(hc_scaled, k = 3)
table(hcRes3, species)
