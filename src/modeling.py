# modeling.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models

# BASE MODELS

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model



def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
    n_estimators=200,      # عدد الأشجار
    max_depth=6,           # أقصى عمق لكل شجرة
    learning_rate=0.1,     # سرعة التعلم
    subsample=0.8,         # أخذ عينات من البيانات لتسريع التدريب وتجنب overfitting
    colsample_bytree=0.8,  # أخذ عينات من المميزات لكل شجرة
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,             # استخدام كل الأنوية المتاحة
    random_state=42
)
    model.fit(X_train, y_train)
    return model


# DEEP LEARNING MODEL
def build_ffnn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# EVALUATION
def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n===== {name} Evaluation =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# SAVE MODEL
def save_model(model, path):
    joblib.dump(model, path)
    print(f"[INFO] Saved: {path}")
