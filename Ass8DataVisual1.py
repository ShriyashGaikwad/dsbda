## Data Visualization I

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = sns.load_dataset('titanic')
print(dataset.head())

dataset = dataset.dropna()

# 1. Distribution Plot (Age)
sns.histplot(dataset['age'], kde=True)
plt.show()

# 2. Box Plot (Gender vs Age)
sns.boxplot(x='sex', y='age', data=dataset)
plt.show()

# 3. Bar Plot (Survival Count)
sns.countplot(x='survived', data=dataset)
plt.show()

