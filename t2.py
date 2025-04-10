Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# Title: Titanic Survival Prediction - Classification Model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as 
plt
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    plt
NameError: name 'plt' is not defined
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    from sklearn.model_selection import train_test_split
ModuleNotFoundError: No module named 'sklearn'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = sns.load_dataset("titanic")
print(df.head())
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True

[5 rows x 15 columns]
print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
None
print(df.describe())
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
df.dropna(inplace=True)
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])
>>> corr = df.corr()
>>> plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
>>> sns.heatmap(corr, annot=True, cmap="coolwarm")
<Axes: >
>>> plt.title("Feature Correlation Matrix")
Text(0.5, 1.0, 'Feature Correlation Matrix')
>>> plt.show()
>>> X = df.drop("survived", axis=1)
>>> y = df["survived"]
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> scaler = StandardScaler()
>>> X_train = scaler.fit_transform(X_train)
>>> X_test = scaler.transform(X_test)
>>> model = RandomForestClassifier(n_estimators=100, random_state=42)
>>> model.fit(X_train, y_train)
RandomForestClassifier(random_state=42)
>>> y_pred = model.predict(X_test)
>>> print("Classification Report:\n", classification_report(y_test, y_pred))
Classification Report:
               precision    recall  f1-score   support

           0       0.79      0.84      0.81        80
           1       0.78      0.71      0.74        63

    accuracy                           0.78       143
   macro avg       0.78      0.78      0.78       143
weighted avg       0.78      0.78      0.78       143

>>> print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
Confusion Matrix:
 [[67 13]
 [18 45]]
>>> print("Accuracy Score:", accuracy_score(y_test, y_pred))
Accuracy Score: 0.7832167832167832
>>> importances = model.feature_importances_
>>> features = X.columns
>>> feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
>>> plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
>>> sns.barplot(x=feat_imp, y=feat_imp.index)
<Axes: xlabel='None', ylabel='None'>
>>> plt.title("Feature Importances")
Text(0.5, 1.0, 'Feature Importances')
>>> plt.show()
