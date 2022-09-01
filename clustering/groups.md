# Clustering 

### Notes from clustering experiments
*  myopia-pca-kMeans.ipynb  
    Unbalanced data
    *  PCA components: 10 (90+% Variance Explained)  
    *  KMeans Clusters: 3 is better than 4 or 5
    *  Tested KNN on first cluster
        *  With K: 3 and oversampling, Cluster 1:  
        *  Accuracy:  64%
        *  Precision: 12%
        *  Recall: 22%
    Note:  A noticable decrease in accuracy compared to KNN with Balanced Data (see above).

    Balanced data with over sampling
    *  PCA components:  


*  myopia-pca-kMeans-KNN.ipynb   

*  myopia-pca-tSNE.ipynb  