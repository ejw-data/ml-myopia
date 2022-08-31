# Finding Significant Features

### Notes from exploratory process
*  Checking each feature based on a ranking method (boxplot) shows that 42% of the 618 records have no outliers in any of the features.  Other features like age shows that most records are with 6 year olds so this analysis will only consider 6 year olds.  I will use this `ideal` dataset initially.
*  After looking at several correlation tests or independence tests to determine relationships to the target value (MYOPIC), the most important features are `SPHEQ`, `SPORTHR`, `DADMY` and `total_positive_screen`.  
*  It should also be noted that there is a strong correlation between ``AL` and `VCD`.  
*  As shown in the VIF analysis, `SPHEQ`, `SPORTHR`, `DADMY` and `total_positive_screen` were evaluated as significant since their VIF scores were all less than 5.  This matches the results of the correlation/independence tests.  
*  Plots indicate that only `SPHEQ` is important to the target, especially when viewed without any filtered data.  
*  It should also be noted that the filtered df has only 34 myopic cases and 233 non-myopic cases.  This is very unbalanced data.  
*  The analysis of feature interactions was done with partial depenence plots (PDP) and Individual conditional expectation (ICE) plots.  Only `AL` showed feature-feature interactions.  `SPHEQ` showed to be influenced the most while the other features showed very little relation to the target.    