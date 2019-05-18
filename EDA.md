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
* 1. Feature Contribution:- It is a way to figure out how much each feature influences the model. 
* 2. _relationship between variables and correlation between features._
* 3. matplotlib and seaborn libraries. 
* 4. Histograms (Seaborn version of histogram is density plot, sns.distplot) and Scatter Plots. Histogram for seeing the distribution of a particular variable, Scatter plot for seeing relationships *  between 2 or more variables.
* 5._Heatmap (in seaborn lib) provides us with a numerical value of the correlation between each variable._
* _Principal Component Analysis (PCA) is used to reduce the number of features to use and graphing the variance which gives us an idea of how many features we really need to represent our dataset fully._
