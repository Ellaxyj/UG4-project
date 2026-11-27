import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

FOLD = 4                                  
FEATURES = ["Boar_density", "Pop_density"]+["tess_" + str(i) for i in range(128)]

#read the model table,2022-2025
table = pd.read_csv("model_table_tessera.csv")
table = table[table["Period"] < "2026"]  
table = table.dropna(subset=FEATURES)
table = table.reset_index(drop=True)

X = table[FEATURES]
y = table["label"]

#split the data into training and test sets using StratifiedGroupKFold
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_splits = list(splitter.split(X, y, groups=table["Province"]))
train_rows = all_splits[FOLD][0]
test_rows = all_splits[FOLD][1]

X_train = X.iloc[train_rows]
X_test = X.iloc[test_rows]
y_train = y.iloc[train_rows]
y_test = y.iloc[test_rows]

#standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

#predict and evaluate
y_predict = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("confusion matrix:")
print(confusion_matrix(y_test, y_predict))

auc = roc_auc_score(y_test, y_prob)
print("\nAUC:", round(auc, 3))

# # ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# plt.figure()
# plt.plot(fpr, tpr, label=f"Fold {FOLD} (AUC = {auc:.2f})")
# plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()