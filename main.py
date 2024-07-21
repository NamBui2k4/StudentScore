import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("StudentScore.xls")

# profile = ProfileReport(data, title="Student Score Report", explorative=True)
# profile.to_file("Statistic.html")

target = "math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
                    "master's degree"]
gender = ["male", "female"]
lunch = x_train["lunch"].unique()
test_prep = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_values, gender, lunch, test_prep])),
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
])

params = {
    "n_estimators": [50, 100, 200],
    "criterion": ['squared_error', 'absolute_error'],
    "max_depth": [None, 2, 5],
    "min_samples_split": [2, 5]
}
from sklearn.linear_model import LinearRegression
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model",LinearRegression())
])

reg.fit(x_train, y_train)

with open('coef.txt', 'w') as f:
    for i,j in zip(x.columns,reg['model'].coef_):
        f.write('{} is within coefficient of: {} \n'.format(i,j))
    

y_pred = reg.predict(x_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
with open('Loss.txt', 'w') as f:
    f.write('mean_absolute_error: {}\n'.format(mean_absolute_error(y_test, y_pred)))
    f.write('mean_squared_error: {}\n'.format(mean_squared_error(y_test, y_pred)))
    f.write('Variance Regression Score: {}\n'.format(explained_variance_score(y_test, y_pred))  )

# visualize results
import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Values')
plt.plot(y_pred, label='Predicted Values')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Math Score')
plt.title('Actual vs Predicted Math Scores')
plt.savefig('Result.png')

# save model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(reg, f)