import pandas as pd
import numpy as np

df = pd.read_csv("academic_performance.csv")
df.head()

df.isnull().sum()

df.describe()

numeric_cols = ['math','science','english','attendance','study_hours']

df['math'] = df['math'].fillna(df['math'].median())
df['science'] = df['science'].fillna(df['science'].median())
df['attendance'] = df['attendance'].fillna(df['attendance'].mean())

df.isnull().sum()

def treat_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return col.clip(lower, upper)

df['math'] = treat_outliers(df['math'])
df['science'] = treat_outliers(df['science'])
df['english'] = treat_outliers(df['english'])
df['attendance'] = treat_outliers(df['attendance'])

df['math_log'] = np.log(df['math'] + 1)
df[['math', 'math_log']].head()

df.head()
df.describe()

numeric_cols = ['math','science','english','attendance','study_hours']

df.head()

df.describe()