# Model Selection

### Notes from model testing experiments
*  model-KNN.ipynb
    Unbalanced Data:
    *  k=5
    *  ROC AUC: 0.77 
    *  Precision: 0.19
    *  Recall: 0.09
    Summary:  The cross validation shows that only a small percent of the predictions are correct.  By using an updated train-test-split with 200 train samples, the precision/recall results decreased slightly.  

    Balanced Data with SMOTE:
    *  k = 5
    *  ROC AUC:  0.75
    *  Precision:  0.28
    *  Recall:  0.65
    Summary:  This model has some predictive power.  It is better at avoiding classification of false negatives. It's still not very good. By changing the train-test-split to have a smaller train set, the results were about the same.     

*  model-SVC-gridsearch.ipynb  
    Unbalanced Data:
    *  Summary:  Training and testing had similar responses for the holdout data but the cross validation scores showed that all the scores were low.  

    Unbalanced Data with GridSearch
    Summary:  I laughed at this result since the optimized value for C was 1 which is the default.  So the result makes sense that the model evaluated to the same values as the non-tuned model.    

    Unbalanced Data with PCA and GridSearch 
    Summary:  PCA worked best with 7 components.  Similar to the above results, PCA indicated to use all the features.  With the training dataset being a bit larger, the model suggests to use 4 components and showed some increases in predictive power.  `This is very interesting and would be a good experiment on synthetically created data.`

    Balanced Data with SMOTE
    Summary:  Saw a significant increase in recall and modest gains in precision relative to the unbalanced experiment.   

    Balanced Data with SMOTE and PCA and Gridsearch
    Summary:  Slight increases in precision but caused only by increasing the SVC C parameter.  

    Balanced Data with SMOTE and SVC class weighting and Gridsearch
    Summary:  The model has a lot of differences between train and test datasets.  This experiment should be repeated with more data.  Weighting had minimal effect on this model; probably because it was not needed due to the balancing step of the model.  

    Balanced Data with SMOTE, PCA, SVC class weighting and Gridsearch
    Summary:  I think this model is overfitting.  I have high train scores but low test scores and the numbers just don't make sense.  I limited the values that the hyperparameters would test with to see if the overfitting could be stopped.  I was able to get some values that made some sense but only by using high C values.  When running a separate cross validation, I was able to get precision and recall values a bit higher than past models.  I'm not sure about this model since I manually forced the C regularization term to be very high.  The class weighting basically did not help in any way since this model is nearly identical to the version without weighting and in some instances I think the weighting is contributing to the overfitting. `Note:  By changing the train-test-split, only small changes occurred.` 

    Unbalanced SVC with Class Weights  
    Summary:  With larger training sets, the results had similar effects as previous models but when reducing the training set I see similar precision results but higher recall results.

    Final Summary:  I think that linear SVC just has a hard time fitting this model well.  The SVC parameter that allows for balancing may be fine to use when that is the only imbalance correcting done but otherwise it does not seem to contribute to the model much.  In general PCA helps a little bit but the biggest contributor to improving the model with this dataset is using oversampling.   

*  model-multimodel-compare.ipynb  
    Model Report indicated these are the best models:  
    *  NearestCentroid
    *  DesicisonTreeClassifier
    *  AdaBoostClassifier
    *  Perceptron
    *  LogisticRegression
    All show 60%+ Accuracy when balanced
    and show 77% Accuracy with unbalanced.  F1 Scores are typically over 80%.  
    My main preference is that it would include the Precision and Recall scores since I like to look at those individually instead of looking at the weighed combination of the two with the f1 score.  The ROC AUC can also be a bit misleading for models that have similar scores since they can have dramatically different precision and recall scores.  I am thinking about including a precision-recall metric to all my evaluations since that may be more helpful with imbalanced data.   

*  model-keras.ipynb 
    Sequential Model with 2 hidden layers with each one having 16 nodes.  
    *  Precision:  0.83
    *  Recall:  0.19
    Although the precision is a bit high, it does seem to have good predictive power and the balance between precision and recall can be modified by changing the threshold criteria.  When changing the threshold criteria I can obtain models that are similar to some of the models where the precision is around 0.47 and recall around 0.35.  

* myopia_lgb.ipynb  
    Using the FLAML AutoML library to get parameters and then using the scikit-learn classifier as a wrapper resulted in precision of 0.55 and recall of 0.34. All the tests results were reasonable and it was nice to see that this model tested similar to the neural network.   