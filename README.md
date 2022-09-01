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

## Process  

Overall I am trying to find the right combination of data preparation, algorithm selection, and algorithm tuning to make the best model possible.  For full details of my process then see the [process writeup](./docs/process.md).

<br>

## Analysis  

### Exploratory Data Analysis

> For more information about the feature selection, see my [notes](./eda/exploratory.md) or read the notes and summaries from the [notebooks](./eda/)

*  For this analysis I decided that it would be fine to work on a highly filtered dataset and then redo the process several times with more variation in the data with each iteration.  
*  The final dataset has data of only 6 year olds and about 260 records after removing records that may be considered outliers (extreme cases), special cases, or faulty data.
*  Of the available features, I selected to use `SPHEQ`, `SPORTHR`, `DADMY` and `total_positive_screen` as signficant features and I also included `ACD`', `LT`, and `VCD` for future experiments.  All the testing I performed indicated that the same varialbes were important.  I should also note that in my future analyses that I used `delta_spheq` instead of `SPHEQ` but the two are very similar.  The label or target is `MYOPIC`.
*  Interacting features could be `AL` since it's partial dependence plot showed some positive and negative results.  `SPHEQ` is alos an interesting variable but it will be left in the dataset.  

### Data Balancing
> The dataset is 13% myopic cases.  This is fairly significantly imbalnced.  

> For more information about the data balancing experiments, see my [notes](./data-balancing/balancing.md) or read the notes and summaries from the [notebooks](./data-balancing/)

__Balancing Effect on Distributions__ 
*  SMOTE or ADASYN seem to work the best when balancing but with this dataset the integrity of the distribution is largely maintained so there are no major distortions when using the other tested methods.  Dataset size is the main issue so undersampling may cause problems.   

__Model Performance using Weighted Targets__ 
*  Oversampling techniques seem to have a greater influence than using the RandomForest balancing feature.  By applying SMOTE or ADASYN and then setting the algorithm to further balance the target if any imbalance exists might better fine tune the model. 

__Classifier Threshold Effect on Precision and Recall__ 
*  For parametric algorithms, adjusting a binary classifiers threshold can have significant effects since it changes the precision and recall balance.  This can be a significant change but needs to be done sensibly and with the context of the problem and goal in mind.   I tested this idea with an SVC algorithm.  This could be a practical step in the model selection step but I would need more data to evaluate its broader reliability (with untrained data).  I should do this test with random forest classifier (even though it is not parametric) so I can compare to other tests I have done; the effect of thresholds should still exist but I need to think about this.

__Imbalance Effects on Tree Models__ 
*  Often tree based models like RandomForests can have siginficant differences compared to the other models.  After looking at four different algorithms with and without SMOTE balancing, I found that BalancedRandomForest to provide good results except for precision.  I question how it's undersampling is effecting the model but there is not an immediate method that exists to access the data it uses after undersampling - I will need to delve into the source code.  This algorithm might be favoring recall intentionally over precision.  The key take-away of this analysis is that trees are sensitive to imbalances and balancing and bagging greatly increase the models predictive abilities.  I would say that I also like RandomForest with SMOTE balancing as a model.   

<br>

### Classification  

> Note:  The original repo contained just a couple models but I started exploring the entire modeling process at a later time and expanded the repo into a multi-part analysis.  I am currently updating the orignal models to utilize some of the information found during the EDA and Balancing experiment.  

> For more information about the models, see my [notes](./classification/models.md) or read the notes and summaries from the [notebooks](./classification/)

*  A variety of models were run and more are being added as time permits.  More models can be found within the `data-balancing` folder.  Those models were specifically testing the effects of imbalanced data.  The contents of this folder are testing a variety of models.
*  As an experiment, I tested the LazyClassifier library but I am not yet confident in its use.  I prefer to use the models directly from scikit-learn since I understand the interworkings better.  This library returns the results of many models in a nice table format.  I use it to guide what models I might want to explore in detail.  The file is named `model-multimodel-compare.ipynb`
*  Other models tested:
    *  KNN
    *  ANN
*  Other techniques applied include:
    *  PCA
    *  Hyperparameter Tuning
*  Overall 

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