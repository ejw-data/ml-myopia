


Ref:  https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio


Steps:
1.  Test variance of the training split
    *  Make a 90/10 split of train to test
    *  Run model on all training data and collect accuracy and other metrics
    *  Split the train set again and again as such:
        *  88.9/11.1  (80% of all data)
        *  87.5/12.5  (70% ..)
        *  85.7/14.3  (60% ..)
        *  83.3/16.7  (50% ..)
2.  Test variance of the testing split
    *  Make a 50/50 split of train to test
    *  Run model on all tresting data and collect accuracy and other metrics
    *  Split the test set again and again as such:
        *  80/20        (40% of all data)
        *  75/25        (30% ..)
        *  66.6/33.4    (20% ..)
        *  50/50        (10% ..)