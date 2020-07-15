import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importing the dataset
df = pd.read_csv('Social_Network_Ads.csv')

# get dummies
df_getdummy = pd.get_dummies(data=df, columns=['Gender'])

# delete column Purchased
X = df_getdummy.drop('Purchased', axis=1)
# index 1 (or 0)
y = df_getdummy['Purchased']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# calculate the accuracy score
# Accuracy = number of times you're right / number of predictions
acc_score = accuracy_score(y_true=y_train, y_pred=classifier.predict(X_train))

print(acc_score)
