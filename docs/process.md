# Machine Learning Process

> Below are the steps that I have thought about when looking at the Myopia dataset.  My overall goal is to go through a variety of machine learning methods in the process of determining a good quality model.   

<br>

The most general description of my process is that I am looking for the best combination of Data Preparation, Learning Algorithm and Hyperparameters that make a representative model.   

<br>

My Process:
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