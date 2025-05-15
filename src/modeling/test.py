import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from predict import predict


df = pd.read_csv("data/processed/Churn.csv")

num_feat = df.drop("Exited", axis=1).select_dtypes(include=["float", "int"]).columns
cat_feat = df.drop("Exited", axis=1).select_dtypes(include=["object"]).columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feat),
        ("cat", OneHotEncoder(), cat_feat),
    ]
)


X = df.drop("Exited", axis=1)
y = df["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_pred = predict(X_test, 0.5)

acc_score = accuracy_score(y_pred=y_pred, y_true=y_test)
cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
class_report = classification_report(y_test, y_pred, output_dict=True)

print(acc_score)
print(cm)
print(class_report)
