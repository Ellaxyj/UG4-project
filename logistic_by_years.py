import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


TEST_PERIOD = "2026-01_to_2026-06"       
FEATURES = ["Boar_density", "Pop_density"]+["tess_" + str(i) for i in range(128)]

#read the model table
table = pd.read_csv("model_table_tessera.csv")
table = table.dropna(subset=FEATURES)


#separate the training set and the test set
train = table[table["Period"] < TEST_PERIOD]
test = table[table["Period"] == TEST_PERIOD]

#filter the features and labels
X_train = train[FEATURES]
y_train = train["label"]
X_test = test[FEATURES]
y_test = test["label"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_predict = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

#confusion matrix
print(confusion_matrix(y_test, y_predict))

auc = roc_auc_score(y_test, y_prob)
print("\nAUC:", round(auc, 3))

# # ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# plt.figure()
# plt.plot(fpr, tpr, label=f"Logistic regression (AUC = {auc:.2f})")
# plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve: predicting " + TEST_PERIOD)
# plt.legend()
# plt.show()