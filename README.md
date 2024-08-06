# House Price Prediction
In this project we will use a multivariable regression and valuation model to predict house prices 

### Table of Contents:
 - [00. Project Overview](#project-overview)
 - [01. Data Exploration and Cleaning](#data-exploration-and-cleaning)
   - [01-1. Preliminary Data Exploration](#preliminary-data-exploration)
 - [02. Data Visualization](#data-visualization)
   - [02-1. House Prices](#house-prices)
   - [02-2. Distance to Employment](#distance-to-employment)
   - [02-3. Number of Rooms](#number-of-rooms)
   - [02-4. Access to Highways ](#access-to-highways )
   - [02-5. Next to the River?](#next-to-the-river?)
 - [03. Understanding the Relationships in the Data](#understanding-the-relationships-in-the-data)
   - [03-1. Pair Plot](#pair-plot)
   - [03-2. More Comparing](#more-compariong)
 - [04. Split Training & Test Dataset](#split-training-and-test-dataset)
 - [05. Multivariable Regression](#multivariable-regression)
   - [05-1. Fisrt Regression](#first-regression)
   - [05-2. Analyse the Estimated Values & Regression Residuals](#analyse-the-estimated-values-&-regression-residuals)
   - [05-3. Data Transformations for a Better Fit](#data-transformations-for-a-better-fit)
 - [06. Regression using Log Prices](#regression-using-log-prices)
 - [07. Compare Out of Sample Performance](#compare-out-of-sample-performance)
 - [08. Predict a Property's Value using the Regression Coefficients](#predict-a-property's-value-using-the-regression-coefficients)
 - [09. Results and Next steps](#results-and-next-steps)
 ---

### Project overview
Real estate company wants to value any residential project before they start. We are tasked with building a model that can provide a price estimate based on a home's characteristics like:
- The number of rooms
- The distance to employment centres
- How rich or poor the area is
- How many students there are per teacher in local schools

In this project we will:
1. Analyse and explore the Boston house price dataset
2. Split our data for training and testing
3. Run a Multivariable Regression
4. Evaluate how our model's coefficients and residuals
5. Use data transformation to improve our model performance
6. Use our model to estimate a property price

###  Data Exploration and Cleaning
#### Preliminary Data Exploration
When we run `df_data.shape`, `df_data.tail()` and `df_data.head()`, we see that there are 506 rows and 14 columns.

![image](https://github.com/user-attachments/assets/55d5e636-9033-469c-9367-41919afd2f6e)
![image](https://github.com/user-attachments/assets/63bebb65-14c3-4dda-87d1-c6e0e1cf6247)

We notice that the columns contain the following information:

- _**CRIM**_     per capita crime rate by town
- _**ZN**_       proportion of residential land zoned for lots over 25,000 sq.ft.
- _**INDUS**_    proportion of non-retail business acres per town
- _**CHAS**_     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- _**NOX**_      nitric oxides concentration (parts per 10 million)
- _**RM**_       average number of rooms per dwelling
- _**AGE**_      proportion of owner-occupied units built prior to 1940
- _**DIS**_      weighted distances to five Boston employment centres
- _**RAD**_      index of accessibility to radial highways
- _**TAX**_      full-value property-tax rate per $10,000
- _**PTRATIO**_  pupil-teacher ratio by town
- _**B**_        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- _**LSTAT**_    % lower status of the population
- _**PRICE**_     Median value of owner-occupied homes in $1000's

We do not need to clean data as there are no duplicates or NaN values

---

### Data Visualization
We will use seaborn to visualize data in our dataset

#### House Prices
```Python
sns.displot(data['PRICE'], 
            bins=50, 
            aspect=2,
            kde=True, 
            color='#2196f3')

plt.title(f'1970s Home Values in Boston. Average: ${(1000*data.PRICE.mean()):.6}')
plt.xlabel('Price in 000s')
plt.ylabel('Nr. of Homes')

plt.show()
```
![image](https://github.com/user-attachments/assets/4d17623a-9a9b-47b6-aee5-59edf9f30927)

#### Distance to Employment
```Python
sns.displot(data.DIS, 
            bins=50, 
            aspect=2,
            kde=True, 
            color='darkblue')

plt.title(f'Distance to Employment Centres. Average: {(data.DIS.mean()):.2}')
plt.xlabel('Weighted Distance to 5 Boston Employment Centres')
plt.ylabel('Nr. of Homes')

plt.show()
```
![image](https://github.com/user-attachments/assets/7e4e130b-4378-4220-8dcf-ffb80c06564a)

#### Number of Rooms
```Python
sns.displot(data.RM, 
            aspect=2,
            kde=True, 
            color='#00796b')

plt.title(f'Distribution of Rooms in Boston. Average: {data.RM.mean():.2}')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Nr. of Homes')

plt.show()
```
![image](https://github.com/user-attachments/assets/12fb0a75-8fdb-4e0b-8315-79c02d3b5928)

#### Access to Highways 
```Python
plt.figure(figsize=(10, 5), dpi=200)

plt.hist(data['RAD'], 
         bins=24, 
         ec='black', 
         color='#7b1fa2', 
         rwidth=0.5)

plt.xlabel('Accessibility to Highways')
plt.ylabel('Nr. of Houses')
plt.show()
```
![image](https://github.com/user-attachments/assets/48bded7c-e1f3-4a68-b8f7-a00586077d1d)

#### Next to the River?
```Python
river_access = data['CHAS'].value_counts()

bar = px.bar(x=['No', 'Yes'],
             y=river_access.values,
             color=river_access.values,
             color_continuous_scale=px.colors.sequential.haline,
             title='Next to Charles River?')

bar.update_layout(xaxis_title='Property Located Next to the River?', 
                  yaxis_title='Number of Homes',
                  coloraxis_showscale=False)
bar.show()
```
![image](https://github.com/user-attachments/assets/ed5c210c-b74f-4989-9cd5-da5ca16e6edf)

---

### Understanding the Relationships in the Data
#### Pair Plot
There might be some relationships in the data that we should know about. We can use Pair Plot to see all relationships among data in the dataset:
```Python
sns.pairplot(data)
plt.show()
```
![image](https://github.com/user-attachments/assets/5fbcaed1-6ba4-47e7-b62f-6cf1a0cdc83b)

#### More Comparing
There are more comparing of the data like:
- Distance from Employment vs. Pollution
- Proportion of Non-Retail Industry vs Pollution
- Number of Rooms versus Home Value
and etc.
You can see them in details in the jupyter notebook file.

---

### Split Training & Test Dataset
We can't use all 506 entries in our dataset to train our model. The reason is that we want to evaluate our model on data that it hasn't seen yet (i.e., out-of-sample data). 
That way we can get a better idea of its performance in the real world.

We can do it in a following order:
- Import the `train_test_split()` function from sklearn
- Create 4 subsets: `X_train`, `X_test`, `y_train`, `y_test`
- Split the training and testing data roughly 80/20.
- To get the same random split every time we run our notebook we use `random_state=10`. This helps us get the same results every time and avoid confusion while we're learning.

```Python
target = data['PRICE']
features = data.drop('PRICE', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=10)
```
```Python
# % of training set
train_pct = 100*len(X_train)/len(features)
print(f'Training data is {train_pct:.3}% of the total data.')

# % of test data set
test_pct = 100*X_test.shape[0]/features.shape[0]
print(f'Test data makes up the remaining {test_pct:0.3}%.')
```

---

### Multivariable Regression
We use sklearn to run the regression on the training dataset.

#### First Regression
```Python
regr = LinearRegression()
regr.fit(X_train, y_train)
rsquared = regr.score(X_train, y_train)

print(f'Training data r-squared: {rsquared:.2}')
```
#### Analyse the Estimated Values & Regression Residuals
The next step is to evaluate our regression. How good our regression is depends not only on the r-squared. It also depends on the residuals - the difference between the model's predictions (
) and the true values (
) inside y_train.

```Python
predicted_values = regr.predict(X_train)
residuals = (y_train - predicted_values)
```
```Python
# Original Regression of Actual vs. Predicted Prices
plt.figure(dpi=100)
plt.scatter(x=y_train, y=predicted_vals, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals vs Predicted values
plt.figure(dpi=100)
plt.scatter(x=predicted_vals, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()
```
![image](https://github.com/user-attachments/assets/540d68e0-8c8c-457d-bddf-b015c77c0b9e)
![image](https://github.com/user-attachments/assets/1be7089c-3e9a-475c-9eba-63cd87b2de10)

#### Data Transformations for a Better Fit

We have two options at this point: 

1. Change our model entirely. Perhaps a linear model is not appropriate. 
2. Transform our data to make it fit better with our linear model. 

Let's try a data transformation approach. 

We have to investigate if the target `data['PRICE']` could be a suitable candidate for a log transformation. 

* Use Seaborn's `.displot()` to show a histogram and KDE of the price data. 
* Calculate the skew of that distribution.
* Use [NumPy's `log()` function](https://numpy.org/doc/stable/reference/generated/numpy.log.html) to create a Series that has the log prices
* Plot the log prices using Seaborn's `.displot()` and calculate the skew. 
* Which distribution has a skew that's closer to zero?

```Python
tgt_skew = data['PRICE'].skew()
sns.displot(data['PRICE'], kde='kde', color='green')
plt.title(f'Normal Prices. Skew is {tgt_skew:.3}')
plt.show()
```
![image](https://github.com/user-attachments/assets/af179add-8f25-4422-8bb4-b2bd01bef861)

```Python
y_log = np.log(data['PRICE'])
sns.displot(y_log, kde=True)
plt.title(f'Log Prices. Skew is {y_log.skew():.3}')
plt.show()
```
![image](https://github.com/user-attachments/assets/c896c1c3-3d37-4e6b-bfe5-3fdac25cdb62)

The log prices have a skew that's closer to zero. This makes them a good candidate for use in our linear model. Perhaps using log prices will improve our regression's r-squared and our model's residuals.

---

### Regression using Log Prices

Using log prices instead, our model has changed to:

$$ \log (PR \hat ICE) = \theta _0 + \theta _1 RM + \theta _2 NOX + \theta_3 DIS + \theta _4 CHAS + ... + \theta _{13} LSTAT $$

```Python
new_target = np.log(data['PRICE']) # Use log prices
features = data.drop('PRICE', axis=1)

X_train, X_test, log_y_train, log_y_test = train_test_split(features, 
                                                    new_target, 
                                                    test_size=0.2, 
                                                    random_state=10)

log_regr = LinearRegression()
log_regr.fit(X_train, log_y_train)
log_rsquared = log_regr.score(X_train, log_y_train)

log_predictions = log_regr.predict(X_train)
log_residuals = (log_y_train - log_predictions)

print(f'Training data r-squared: {log_rsquared:.2}')
```
And if we compare mean and skew values of two models it is obvious that the second model is more accurate:
```Python
log_resid_mean = round(log_residuals.mean(), 2)
log_resid_skew = round(log_residuals.skew(), 2)

sns.displot(log_residuals, kde=True, color='navy')
plt.title(f'Log price model: Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')
plt.show()

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Original model: Residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()
```
![image](https://github.com/user-attachments/assets/2b48f55d-5b01-4696-a783-12c09b8af29c)  
![image](https://github.com/user-attachments/assets/e01f0061-e0fb-41d8-bb8e-0b0bd0861ce3)

Our new regression residuals have a skew of 0.09 compared to a skew of 1.46. The mean is still around 0. From both a residuals perspective and an r-squared perspective we have improved our model with the data transformation

---

### Compare Out of Sample Performance
The real test is how our model performs on data that it has not "seen" yet. This is where our `X_test` comes in.
```Python
print(f'Original Model Test Data r-squared: {regr.score(X_test, y_test):.2}')
print(f'Log Model Test Data r-squared: {log_regr.score(X_test, log_y_test):.2}')
```

By definition, the model has not been optimised for the testing data. Therefore performance will be worse than on the training data. However, our r-squared still remains high, so we have built a useful model.

---

### Predict a Property's Value using the Regression Coefficients
Let's predict how much the average property is worth using the stats above.
```Python
# Make prediction
log_estimate = log_regr.predict(property_stats)[0]
print(f'The log price estimate is ${log_estimate:.3}')

# Convert Log Prices to Acutal Dollar Values
dollar_est = np.e**log_estimate * 1000
# or use
dollar_est = np.exp(log_estimate) * 1000
print(f'The property is estimated to be worth ${dollar_est:.6}')
```
A property with an average value for all the features has a value of $20,700.

---

### Results and Next Steps
Overall our model can predict house price values depending on the features of the house. However, to make it more accurate we could:
- Try better models to see if they predict more accurately;
- Try better data transformations
- Use a bigger dataset to train our model 
