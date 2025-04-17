
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Để đánh giá và so sánh các mô hình
from sklearn.model_selection import cross_validate
df = pd.read_csv('Bank Customer Churn Prediction.csv')
df = df.dropna()
data = df.drop(['customer_id'], axis = 1)

# One-hot encode the categorical columns
data = pd.get_dummies(data, columns=['country', 'gender', 'credit_card', 'active_member'], drop_first=True)
data = data.astype(int)
numerical_columns = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

X = data.drop(['churn'],axis=1)
y = data['churn']
X_res,y_res = SMOTE().fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2,random_state=42)

results = {}

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}


for model_name, model in models.items():
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    results[model_name] = {
        'accuracy': np.mean(scores['test_accuracy']),
        'precision': np.mean(scores['test_precision_macro']),
        'recall': np.mean(scores['test_recall_macro']),
        'f1': np.mean(scores['test_f1_macro'])
    }

# Chuyển kết quả thành DataFrame để dễ so sánh
results_df = pd.DataFrame(results).T

best_model = results_df['f1'].idxmax()
print(f"Best model based on F1-score: {best_model}")
print(results_df)

best_model_instance = models[best_model]
best_model_instance.fit(X_train, y_train)
y_pred = best_model_instance.predict(X_test)

final_results = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='macro'),
    'recall': recall_score(y_test, y_pred, average='macro'),
    'f1': f1_score(y_test, y_pred, average='macro')
}

print(f"Final results on test set for the best model ({best_model}):")
print(final_results)
