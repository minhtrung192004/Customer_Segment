import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def prepare_pipeline(numeric_features, categorical_features):
    """
    Tạo pipeline xử lý dữ liệu với chuẩn hóa số và mã hóa danh mục
    """
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    return preprocessor


def split_data(df, target_column, test_size=0.2):
    """
    Tách tập train/test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_model(model, param_grid, X_train, y_train, search_type='grid', cv=5):
    """
    Huấn luyện mô hình với GridSearchCV hoặc RandomizedSearchCV
    """
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, n_iter=30)
    search.fit(X_train, y_train)
    return search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    In báo cáo đánh giá mô hình
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        print("AUC Score:", auc)