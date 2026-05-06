import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df.head()

df.shape
df.info()

df.describe()
df.isnull().sum()

df.dtypes
df['Survived'] = df['Survived'].astype('category')

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

df = pd.get_dummies(df, columns=['Embarked'])

df['Cabin'] = df['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)

df[['Embarked_C', 'Embarked_Q', 'Embarked_S']] = df[['Embarked_C', 'Embarked_Q', 'Embarked_S']].astype(int)

df.head()

df.info()