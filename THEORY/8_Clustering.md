
### 8. Clustering

**K-Means Clustering**
- Unsupervised Machine Learning algorithm.
- Groups similar data points into K clusters.
- Each cluster has a centroid.
- Assigns data points to the nearest centroid.
- Python: `KMeans()`, `fit_predict()`

**Centroid**
- Central point of a cluster.
- Calculated as the mean of all points in the cluster.
- Updated during each K-Means iteration.
- Represents the center of a cluster.

**Euclidean Distance**
- Measures straight-line distance between two points.
- Used to assign data points to the nearest centroid.
- Formula: `d = √[(x₂-x₁)² + (y₂-y₁)²]`
- Smaller distance indicates greater similarity.

**Elbow Method**
- Used to find the optimal number of clusters (K).
- Calculates WCSS (Within-Cluster Sum of Squares).
- Plots K against WCSS.
- The elbow point indicates a suitable K value.
- Python: `model.inertia_`

**Hierarchical Clustering**
- Unsupervised Machine Learning algorithm.
- Creates a hierarchy of clusters.
- Agglomerative: Bottom-up approach (merging clusters).
- Divisive: Top-down approach (splitting clusters).
- Python: `AgglomerativeClustering()`

**Dendrogram**
- Tree-like diagram representing hierarchical clustering.
- Shows how clusters are merged or split.
- Height represents linkage distance.
- Used to select the number of clusters.
- Python: `scipy.cluster.hierarchy.dendrogram()`
