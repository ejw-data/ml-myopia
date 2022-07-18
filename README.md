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
*  Imbalanced-Learn

<br>


## Data Source  
Reduced dataset from [Orinda Longitudinal Study of Myopia conducted by the US National Eye Institute](https://clinicaltrials.gov/ct2/show/NCT00000169)

<br>

## Analysis  
To Be Added Soon...

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

