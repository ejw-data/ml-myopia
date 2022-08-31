# Compensating for Data Imbalances

### Notes from exploratory process
*  Probably due to the outlier filtering performed in the Exploratory Data Analysis step, most balancing techniques did not distort the distribution significantly and most balanced datasets performed similarly when looking at the cross validation calculations.   
*  I need to compare the unfiltered dataset to see how much balancing effects the distribution to better understand the limits of balancing.  
*  Some skepticism exists if the holdout dataset is large enough to make a valid prediction and evaluation.
*  In summary, most of the models performed very similarly.  The two best balancing methods seemed to be ADASYN and SMOTE.  Both methods resulted in models with an ROC AUC of ~0.66; recall of ~0.59, and precision of 0.19.  
*  In the future, I might want to add the standard deviation to the cross validation metrics and also add the precision-recall AUC as an additional metric for evaluation.       
*  An example of how the threshold in the classifier can be changed from 50% to other values to modify the cross validation results.  For right now this problem can keep the standard threshold.  
*  Tests were also run to see the effect of using weights on a RandomForest model.  It appears that balancing the data has much more significant results and that weights could be used to fine tune the model.  
*  BalancedRandomForest performed the best with only precision receiving lower scores.  A model that could be useful for predicting with high precision is the RandomForests with SMOTE or AdaBoost.  
* Modifying the theshold was done using the SVC algorithm and using the .predict_proba() function and calculating the new cutoff point.  This showed significant changes in the model.  ADASYN and Smote with SVC performed the best but not as well as some of the other models.  This process would be best to do during parameter tuning.  