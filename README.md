# ml-clustering-myopia

Author:  Erin James Wills, ejw.data@gmail.com  

![Myopia Banner](./images/myopia-analysis-ml.png)  

<cite>Photo by <a href="https://unsplash.com/@v2osk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">v2osk</a> on <a href="https://unsplash.com/s/photos/eye?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a></cite>

<br>

## Overview  
<hr>  

Clustering used to identify nearsighted patients  

This dataset has actually had multiple publications over the past few years and it appears that the abstracts determine that the data provided in the full dataset has very limited predictive abilities.  After looking at the subset of data, I would agree and one of the main issues with my analysis revolves around the dataset being too small.  

In my analysis, I tested several different methods.  Initially, I just ran a KNN model to see its predictive value.  The model was not able to predict positive myopia cases.  By oversampling, the data so the dataset was more balanced between myopia positive and negative patients, I was able to get some predicive ability from the model.  I continued to use pca and t-SNE to determine if there were some relationships - some patterns were observeable.  

As the last exploratory task, I decided to see if I could run a kMeans algorithm on the data and deterine if any of the groupings could provide additional predictive value.  At first it seemed promising but upon changing the random_state of the datasplits, the results change significantly; leading me to think the data set is too small to get a representative training sample.   
   
<br>  

## Technologies    
*  Python
*  Scikit-Learn
*  TensorFlow
*  Imbalanced-Learn

<br>


## Data Source  
Reduced dataset from [Orinda Longitudinal Study of Myopia conducted by the US National Eye Institute](https://clinicaltrials.gov/ct2/show/NCT00000169)

<br>

## Analysis  

### Classification  
*  myopia-KNN.ipynb
    Unbalanced Data:
    *  k=5
    *  Accuracy: 88% 
    *  Precision: 0%
    *  Recall: 0%
    Summary:  This model predicts everything as False and has no predictive power.  The reason why it predicts this way is becasue the data is unbalanced.  

    Balanced Data with Oversampling:
    *  k = 7
    *  Accuracy:  76%
    *  Precision:  23%
    *  Recall:  44%
    Summary:  This model has some predictive power.  It correctly predicts negative results 92% of the time and positive results 23% of the time.  

*  myopia-gridsearch-SVC-pipeline.ipynb  
    Unbalanced Data:
    *  PCA Components:  10
    *  Accuracy: 89%
    *  Precision:  57%
    *  Recall:  22%
    *  Summary:  This model does a much better job at predicting.  

    Balanced Data with Oversampling
    *  PCA Components:  10
    *  Accuracy:  88%
    *  Precion:  0%
    *  Recall: 0%
    Summary:  Balancing followed by PCA reduced predictive ability.  

*  myopia-multimodel-compare.ipynb  
    Model Report indicated these are the best models:  
    *  NearestCentroid
    *  DesicisonTreeClassifier
    *  AdaBoostClassifier
    *  Perceptron
    *  LogisticRegression
    All show 60%+ Accuracy when balanced
    and show 77% Accuracy with unbalanced.  F1 Scores are typically over 80%.  

*  myopia-keras.ipynb 
    Sequential Model with 2 hidden layers with each one having 16 nodes.  
    *  Accuracy:  86%  
    *  Precision:  40%
    *  Recall:  10%

### Clustering

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

<br>

## Setup and Installation  
1. Environment needs the following:  
    *  Python 3.6+  
    *  pandas  
    *  scikit-learn
    *  imb-learn
    >To install imb-learn:  `conda install -c conda-forge imbalanced-learn`
1. Clone the repo to your local machine
1. Activate your environment in that directory  
1. Open a Jupyter Notebook   
1. Run any of the following notebooks:  
    *   `myopia-KNN.ipynb` 
    *   `myopia-pca-kMeans.ipynb`
    *   `myopia-pca-tSNE.ipynb`
    *   `myopia-pca-kMeans-KNN.ipynb`

## Create environment for LazyClassifier
In gitbash or terminal, perform the following commands:
1.  Type `conda create -n PythonML python=3.6 anaconda`
1.  Type `conda install -c anaconda nb_conda_kernels`
1.  Type `conda install -c conda-forge/label/cf202003 nodejs`
1.  Type `pip install scikit-learn==0.23.1`
1.  Type `pip install sklearn-utils`
1.  Type `pip install --upgrade pyforest`
1.  Type `python -m pyforest install_extensions`
1.  Type `pip install lazypredict`  

Note:  I am using Jupyter Lab (might be the reason step 3 is needed)


## Examples  

![Compare Models](./images/models_compare.png)