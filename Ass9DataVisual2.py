## Title Data Visualization II

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#titanic dataset
data = pd.read_csv("titanic_train.csv")

# 1. Pie Chart of Sex
data['Sex'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

# 2. Distplot of Age
sns.histplot(data['Age'].dropna(), kde=True)
plt.show()

# 3. Crosstab & Heatmap (Pclass vs Survived)
ct = pd.crosstab(data['Pclass'], data['Survived'])
print(ct)
sns.heatmap(ct, annot=True, cmap="YlGnBu")
plt.show()
