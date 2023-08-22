import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings('ignore')

# Try different encodings until you find the correct one
encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'utf-16', 'utf-32']


df = None
for encoding in encodings:
    try:
        print(f"Trying encoding: {encoding}")
        df = pd.read_csv('sth.csv', encoding=encoding)
        break
    except (UnicodeDecodeError, pd.errors.ParserError):
        print(f"Failed with encoding: {encoding}")
        continue

if df is None:
    raise ValueError("Failed to read the CSV file with all attempted encodings.")

# Print the first few rows of the DataFrame to inspect its content
print(df.head())



# Rest of your code...


df = pd.read_csv('sth.csv')
#print(df)

Y = df['Species']

X = df.drop(['Species','Id'], axis=1)
#print(X)
#print(X.shape)

#print(Y)
#print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#print(X_train.shape)

#====KNN=======


lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)

lr_cm = confusion_matrix(y_test,lr_pred)
print(lr_cm)

lr_acc = accuracy_score(y_test,lr_pred)*100
print('Accuracy of our model is equal ' + str(round(lr_acc, 2)) + '%')


pickle.dump(lr, open('lr_classifier.pkl','wb'))
