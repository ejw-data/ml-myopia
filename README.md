# ml-clustering-myopia

Author:  Erin James Wills, ejw.data@gmail.com  

![Myopia Banner](./images/myopia-analysis-ml.png)  

<cite>Photo by <a href="https://unsplash.com/@v2osk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">v2osk</a> on <a href="https://unsplash.com/s/photos/eye?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a></cite>

<br>

>Note:  This repo is used for exploring different machine learning techniques. The dataset was selected due to the difficulty of developing a solid model without inappropriately coercing the data.  This is an ongoing project.  

## Overview  
<hr>  

This repo explores multiple machine learning techniques to identify nearsighted patients.  

This dataset is a subset of the `Orinda Longitudinal Study of Myopia (OLSM)` dataset which observed children starting from about 5 to 9 years old for a period of six years.  The data in the subset appears to be data collected upon their initial enrollment in the study and whether the subject developed myopia during the 6 year timeframe.  Myopic was defined as when a subject has a spherical equivalent refraction (SPHEQ) measurement of less than -0.75D.  It should be noted that initial SPHEQ was recorded but changes in this variable over time were not included nor was the date when the subject was verified as having myopia.  

Having the entire dataset covering the entire range of measurements would be interesting especially when determining the rate of change in SPHEQ.  Although this is an interesting problem, analyzing the data over time would probably be a better method than the snapshot in time provided.  

**I chose this dataset to apply to machine learning due to several factors.  First, I know the answer is not clearcut (due to inherent latent factors, dataset size, dataset quality, dataset consistency, etc.) and requires a significant effort to explore the data.  Second, myopia has been studied quite a bit even in 2020 and the predictive value from varying datasets is not very good. Some of the best studies have derived an AUC between 0.801 and 0.837[1] for predicitng myopia progression.**  

It is difficult to say absolutely but this subset of data appears to have been used in multiple publications over the past few years and may exist as a case study in some textbooks; although, I do not have direct access to verify.  What I have seen is questionable applications of models or at least questionable methods of reporting of the results.  Which makes me interested in analyzing the problem and applying different machine learning techniques.  

[1] <cite>Zhang C, Zhao J, Zhu Z, Li Y, Li K, Wang Y, Zheng Y. Applications of Artificial Intelligence in Myopia: Current and Future Directions. Front Med (Lausanne). 2022 Mar 11;9:840498. doi: 10.3389/fmed.2022.840498. PMID: 35360739; PMCID: PMC8962670. Url: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8962670/</cite>  

   
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

Process:
1.  Start with a Goal
1.  Data Search
1.  Exploratory Data Analysis
1.  Analyze the Ask
1.  Set Priorities and Limits
1.  Select Intial Process (Model)
1.  Re-evaluate the Goal and Value
1.  Preprocess Data for Algorithm
    *  Remove Unnecessary Features
    *  Remove Outliers
    *  Replace Missing Values
    *  Balance Data
    *  One-hot-encode Feature Classes
    *  Label-encode Target Classes
    *  Transform Data (for parametric algorithms)
    *  Scale Data (distance or gradient descent or regularized algorithms)
1.  Run Model 
1.  Evaluate Models
    *  Classification Reports
    *  ROC AUC or Precision-Recall AUC
1.  Adjust Models
    *  Changing Theshold Effects
    *  Tuning Parameters
    *  Remove Features
1.  Select Best Model
1.  Re-evaluate the Ask?



### Exploratory Data Analysis
*  Checking each feature based on a ranking method (boxplot) shows that 42% of the 618 records have no outliers in any of the features.  Other features like age shows that most records are with 6 year olds so this analysis will only consider 6 year olds.  I will use this `ideal` dataset initially.
*  After looking at several correlation tests or independence tests to determine relationships to the target value (MYOPIC), the most important features are `SPHEQ`, `SPORTHR`, `DADMY` and `total_positive_screen`.  
*  It should also be noted that there is a strong correlation between ``AL` and `VCD`.  
*  As shown in the VIF analysis, `SPHEQ`, `SPORTHR`, `DADMY` and `total_positive_screen` were evaluated as significant since their VIF scores were all less than 5.  This matches the results of the correlation/independence tests.  
*  Plots indicate that only `SPHEQ` is important to the target, especially when viewed without any filtered data.  
*  It should also be noted that the filtered df has only 34 myopic cases and 233 non-myopic cases.  This is very unbalanced data.  
*  The analysis of feature interactions was done with partial depenence plots (PDP) and Individual conditional expectation (ICE) plots.  Only `AL` showed feature-feature interactions.  `SPHEQ` showed to be influenced the most while the other features showed very little relation to the target.     

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