# EDA : Some insights

* https://www.kaggle.com/ekami66/detailed-exploratory-data-analysis-with-python
* histogram and estimated probability density functions (numerical variables) 
* histogram - value_count() and bar.plot() (categorical variables)
* library seaborn (sns.pairplot() , scatter plot , sns.distplot() to show skew)
* outlier detection (boxplot())
* correlation matrix (sns.heatmap(corr())


# EDA checklist explained

* Q1. What question are you trying to solve (or prove wrong)? [Start with simplest hypothesis]
* Q2. What kind of data do you have? [Numerical, Categorical, Other. How to deal with it?]
*    (Understanding the data)
* Q3. What is missing from the data? And how to deal with it? [avg., replacing with some value, dropping the entire column if not imp,    etc.]
*    (Missing Values)
* Q4. What are the potential outliers? Why should you pay attention to it? [What are they? Do we need them? are they destroying model?]
*    (Plot the distribution of features.)
* Q5. How can you add, remove or change features to get more out of the data? [thumb rule: more data = good]
*    (Feature Engineering. This also includes converting categorical to numerical data.)
* Feature Contribution:- It is a way to figure out how much each feature influences the model. 
