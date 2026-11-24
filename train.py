"""RoomGuard training - Decision Tree, split 70/15/15 stratified, seed 42."""
import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report)
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
BASE = Path.home()/"workspace/roomguard"
df = pd.read_csv(BASE/"data/processed/dataset_roomguard_2000.csv")
X = df[["Light_Lux","Sound_dB"]]; y = df["Occupancy_Binary"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
print(f"train {len(X_train)} val {len(X_val)} test {len(X_test)}")
print("train dist", y_train.value_counts().to_dict(), "val", y_val.value_counts().to_dict(), "test", y_test.value_counts().to_dict())

param_grid = {"max_depth":[3,4,5,6,7,8,None], "min_samples_leaf":[5,10,20], "criterion":["gini","entropy"]}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv, scoring="accuracy")
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_, "CV acc:", round(gs.best_score_,4))
model = gs.best_estimator_
# validate on val, then final test (test tidak pernah dipakai untuk memilih model)
for name, Xt, yt in [("train",X_train,y_train),("validation",X_val,y_val),("test",X_test,y_test)]:
    p = model.predict(Xt)
    print(name, "acc", round(accuracy_score(yt,p),4), "prec", round(precision_score(yt,p),4),
          "rec", round(recall_score(yt,p),4), "f1", round(f1_score(yt,p),4))
    if name=="test":
        print(classification_report(yt,p,target_names=["Kosong","Terisi"]))
        cm = confusion_matrix(yt,p)
        print("CM [[TN,FP],[FN,TP]]\n", cm)
        # also per-source test accuracy
        df_test = df.loc[X_test.index].copy(); df_test["pred"]=p; df_test["correct"]=(df_test["pred"]==df_test["Occupancy_Binary"])
        print(df_test.groupby("Source")["correct"].agg(["mean","count"]))
        print(df_test.groupby(["Source","Occupancy"])["correct"].mean())

# Save model + metrics
(BASE/"models").mkdir(exist_ok=True)
joblib.dump(model, BASE/"models/decision_tree_roomguard.pkl")
# feature importance
imp = dict(zip(["Light_Lux","Sound_dB"], [round(float(v),4) for v in model.feature_importances_]))
print("Feature importance:", imp)
print("\nTree rules:\n", export_text(model, feature_names=["Light_Lux","Sound_dB"], class_names=["Kosong","Terisi"]))
metrics = {}
for name, Xt, yt in [("train",X_train,y_train),("validation",X_val,y_val),("test",X_test,y_test)]:
    p=model.predict(Xt)
    metrics[name]={"n":int(len(yt)),"accuracy":round(float(accuracy_score(yt,p)),4),
      "precision":round(float(precision_score(yt,p)),4),"recall":round(float(recall_score(yt,p)),4),
      "f1":round(float(f1_score(yt,p)),4)}
metrics["best_params"]= {k:(v if v is not None else "None") for k,v in gs.best_params_.items()}
metrics["cv_accuracy"]=round(float(gs.best_score_),4)
metrics["feature_importance"]=imp
metrics["confusion_matrix_test"]=confusion_matrix(y_test, model.predict(X_test)).tolist()
metrics["split"]={"train":1400,"validation":300,"test":300}
metrics["dataset"]={"total":2000,"public":1500,"private":500}
(BASE/"models/metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# Plots
plt.figure(figsize=(6,5)); sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt="d",
  cmap="Blues", xticklabels=["Kosong","Terisi"], yticklabels=["Kosong","Terisi"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - Test Set (n=300)")
plt.tight_layout(); plt.savefig(BASE/"models/confusion_matrix.png", dpi=200); plt.close()
plt.figure(figsize=(14,8)); plot_tree(model, feature_names=["Light_Lux","Sound_dB"], class_names=["Kosong","Terisi"], filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree - RoomGuard (Light_Lux, Sound_dB)"); plt.tight_layout()
plt.savefig(BASE/"models/decision_tree.png", dpi=200); plt.close()
# scatter
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="Light_Lux", y="Sound_dB", hue="Occupancy", alpha=.55, s=18)
plt.title("Light (Lux) vs Sound (dB) by Occupancy - 2000 rows"); plt.tight_layout()
plt.savefig(BASE/"models/scatter_light_sound.png", dpi=200); plt.close()
print("saved model + plots")
