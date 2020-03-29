#Linear Regression Project
#Purpose of this project is to use a linear regression model to identify if a company should focus more on development of their mobile app, or their website.
#Sample data set: Ecomerce Customers


#Imports
import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Loading Data
customers = pd.read_csv("Ecommerce Customers")

#Getting Familiar with the dataset
customers.head()
customers.describe()
customers.info()
#From this, we will be using the "Avg Session Length", "Time on App", "Time On Website", and "Length of Membership" to predict "Yearly Amount Spent"


#Graphically identying where correlation could occur.
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)
# No Great correlation

sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)
#Better Correlation

#Using Pair Plot to view all examples for best correlation with Yearly amount spent
sns.pairplot(data=customers)
#From this, I would guess the best predictive field for Yearly Amount Spent are (in order): Lenth of memberhsip, Time on App, Avg Session Length, Time on Website



#Training and Testing Data
#Splitting numerical features into X, and "Yearly Amount Spent" into y

X = customers[['Avg. Session Length',
 'Time on App',
 'Time on Website',
 'Length of Membership']]
y = customers["Yearly Amount Spent"]

#Using train_test_split from skilearn to split 30% of data into test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


#Training the model using Linear Regression
#Creating an instance of Linear Regression
lm = LinearRegression()

#Train/fit lm on the training data
lm.fit(X_train,y_train)

#Predicting Test Data
predicted = lm.predict(X_test)

#Quick graphic check of real values vs predcited
mpl.scatter(y_test, predicted)
#Looks like a decent model, now to evaluate using MAE, MSE, RMSE

#Printing out errors
print('MAE:', metrics.mean_absolute_error(y_test, predicted))
print('MSE:', metrics.mean_squared_error(y_test, predicted))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
#Errors are reasonable


#Further, we can check the resisuals to make sure everything is in line:
sns.distplot(y_test-predicted)


#Exploring Cofficients:
#Now that the model is well fitted, we can answer the original question, should the company focus more on their website, or the app?
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


#Conclusion
#You can look at the original question in two ways:
#First, the Time on App has a much more significant effect on Yearly Amount Spent. The company could develop the app further to try and raise this number even more.
#Or, the company could recognize that their website really doesn't change Yearly Amount Spent. Instead of focusing on the app, they could spend more time fixing the website to raise this number.











