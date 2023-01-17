##Importing file and organizing columns

#Importing libraries
import pandas as pd

# from sklearn.model_selection import train_test_split

#Importing dataset

df = pd.read_csv('./data.txt', header=None, encoding='utf-8')

#Renaming columns
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

#Mapping output values to int
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2})

#Printing out pandas dataframe
df

#Defining input and target variables for both training and testing
X = df.iloc[:100,[0,1,2,3]].values
y = df.iloc[:100,[4]].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)