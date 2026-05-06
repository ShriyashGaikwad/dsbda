
# DSBDA Exam Survival Guide & Memory Tricks

Don't panic! 10 practicals might sound like a lot, but they actually all follow the exact same patterns. You don't need to memorize 10 different scripts; you just need to memorize **4 core formulas**. 

Here is the best trick to memorizing your DSBDA practicals:

## 1. The Machine Learning Formula (Practicals 4, 5, 6)
Every single ML algorithm (Linear Regression, Logistic Regression, Naive Bayes) uses the **exact same 5-step Sklearn formula**. If you memorize these 5 lines, you know all three practicals:

**Trick:** Just remember the acronym **S.I.T.P.E** *(Split, Initialize, Train, Predict, Evaluate)*. The only thing that changes between practicals 4, 5, and 6 is the name of the model you import!

1. **Split Data:** `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`
2. **Initialize:** `model = LinearRegression()` *(or `LogisticRegression()`, or `GaussianNB()`)*
3. **Train:** `model.fit(X_train, y_train)`
4. **Predict:** `y_pred = model.predict(X_test)`
5. **Evaluate:** `accuracy_score(y_test, y_pred)` *(or `mean_squared_error` for Linear Regression)*

## 2. The Data Visualization Formula (Practicals 8, 9, 10)
All plots use the same skeleton. You just change the function name.

**Trick:** If you forget a complicated plot during the exam, just fall back to a simple `sns.histplot()` or `sns.boxplot()` and `plt.show()`. They are the safest bets and easiest to type.

1. **Load data:** `data = pd.read_csv(...)` or `sns.load_dataset(...)`
2. **Handle nulls:** `data = data.dropna()`
3. **Plot:** 
   - `sns.histplot(data['column'])` (For 1 variable: Distribution)
   - `sns.boxplot(x='col1', y='col2', data=data)` (For 2 variables: Outliers/Comparison)
   - `sns.countplot(x='col1', data=data)` (To count categories)
4. **Show:** `plt.show()`

## 3. The Data Wrangling & Stats Formula (Practicals 1, 2, 3)
This is all pure Pandas. Memorize these 5 "magic cheat codes" for Dataframes:

**Trick:** Whenever you are asked to clean data, always start by running `.isnull().sum()`. It shows the examiner you know the first step of data science.

* **Find empty values:** `df.isnull().sum()`
* **Fill empty values:** `df['Age'].fillna(df['Age'].mean())`
* **Change data types:** `df['Age'].astype('int')`
* **Categorical to Numbers:** `pd.get_dummies(df['Gender'])`
* **Statistics:** `df.groupby('Species').agg(['min', 'max', 'mean'])`

## 4. The Text Analytics / NLP Pipeline (Practical 7)
Treat NLTK like an assembly line for text. Memorize this sequence: **T.S.L.**

1. **T**okenize (Break into words): `word_tokenize(text)`
2. **S**topwords (Remove filler like "the", "and"): `[word for word in tokens if word not in stopwords.words('english')]`
3. **L**emmatize (Convert to root word - e.g., 'running' -> 'run'): `WordNetLemmatizer().lemmatize(word)`

---

## 💡 Quick Tips for the Exam:
1. **Don't memorize spelling:** Learn the *logic*. VS Code/Jupyter has autocomplete. If you type `train_t...` and press Tab, it will write `train_test_split` for you. 
2. **Focus on the Imports:** Often, knowing what to import helps you remember the code. 
   - ML: `from sklearn.model_selection import train_test_split`
   - Graphs: `import seaborn as sns` and `import matplotlib.pyplot as plt`
3. **Keep it minimal:** Don't write extra code if it's not strictly asked for. Stick to what we put in the `DSBDA_mini` files.
