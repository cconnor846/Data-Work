#Logistic Regression Project
#Purpose of this project is to use a logistical regression model to predict if a customer will clik on an advertisment based on thier usage and demographic features.
#Sample data set: advertising

#Imports
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Loading Data
ad_data = pd.read_csv("advertising.csv")

#Getting Familiar with the dataset
ad_data.head()
ad_data.describe()
ad_data.info()
#From this, some text based fields will probably be ignored (such as "Ad Topic Line", "City" and "Country")

#Exploring the dataset to try and discover patterns that could help the model

#Looking at "Age"
sns.distplot(ad_data["Age"])

#Age is probably a good indicator. Area income would also be a good indicator as richer people would probably buy more, and click on more ads.
sns.jointplot(x="Area Income", y="Age", data = ad_data)
#This chart is interesting, as it shows age may not as important

#Another leading indicator is probably time spent on site:
sns.jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data, kind="kde", color="red")
#Younger people are spending more time on site, and they also apper to have a higher area income

#Finally, you can explore all options with pairplot. We will add the hue="Clicked on Ad" to help recognize where the correlations are:
sns.pairplot(ad_data, hue="Clicked on Ad")

#Training and Testing Data
#Exploring the data, we want to use all numerical fields as our X, and "Clicked on Ad" as our y
X_train, X_test, y_train, y_test = train_test_split(ad_data.drop(["Ad Topic Line", "City","Country", "Timestamp"], axis=1), ad_data["Clicked on Ad"], test_size=0.30, random_state=101)

#Creating Logistic Regression instance:
logmodel = LogisticRegression()

#Training and Fitting data on model:
logmodel.fit(X_train,y_train)

#Predictions values for the testing data:
predictions = logmodel.predict(X_test)

#Viewing Classification report to evaluate model:
print(classification_report(y_test,predictions))

#.91 avg across scores is pretty decent. Maybe this could be improved by data mining the text fields, but for a baisc example this model could work.

#Viewing results in confusion matrix:
confusion_matrix(y_test, predictions)
#Again these are decent results. The model limits type 1 and type 2 errors. While correclty predicting who clicked and who did not click on ads.
#The modes can now be used to predict future customers.
