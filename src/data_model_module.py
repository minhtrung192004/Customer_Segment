def data_model_main():
    
    # In[47]:
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    
    from scipy import stats
    from scipy.stats import skew
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, Normalizer
    from sklearn.impute import KNNImputer
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    
    from sklearn.compose import ColumnTransformer
    from imblearn.pipeline import Pipeline  
    from imblearn.over_sampling import SMOTE
    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.multiclass import OneVsRestClassifier
    from datetime import datetime as dt
    import warnings
    
    from scipy.stats import randint, loguniform
    from scipy.stats import uniform
    from sklearn.model_selection import StratifiedKFold
    
    warnings.filterwarnings('ignore')
    
    
    # In[3]:
    
    
    #Define palette
    pal = ["#274D60","#6BA3BE","#0C969C","#0A7075","#032F30", "#031716"]
    cmap = ListedColormap(pal)
    palette = pal
    
    from matplotlib.colors import LinearSegmentedColormap
    pal_1 = ["#032F30","#0A7075","#0C969C","#fff9e6","#ffebc6","#fcd997","#f5b971"]
    palette_1 = pal_1
    
    reversed_pal_1 = pal_1[::-1]
    
    cmap_1 = LinearSegmentedColormap.from_list("reversed_blue_green_cmap", reversed_pal_1)
    
    
    # In[4]:
    
    
    plt.rcParams['font.family'] = 'Times New Roman'
    
    
    # In[5]:
    
    
    transaction_df=pd.read_csv('/Users/hominhtrung/Documents/Giáo Trình - FTU2/Năm 3/Phân tích dữ liệu/[PTDL] Final Project/transaction.csv', index_col=0)
    customer_df=pd.read_csv('/Users/hominhtrung/Documents/Giáo Trình - FTU2/Năm 3/Phân tích dữ liệu/[PTDL] Final Project/customer_data_segmented.csv', index_col=0)
    newwcustomer_df=pd.read_csv('/Users/hominhtrung/Documents/Giáo Trình - FTU2/Năm 3/Phân tích dữ liệu/[PTDL] Final Project/newcustomerlist.csv', index_col=0)
    
    
    # In[6]:
    
    
    customer_df
    
    
    # # 1. Import Data
    
    # In[7]:
    
    
    train = pd.merge(transaction_df, customer_df, how='inner', left_on='customer_id', right_on='customer_id')
    
    
    # In[8]:
    
    
    train
    
    
    # In[9]:
    
    
    test=newwcustomer_df
    
    
    # In[10]:
    
    
    pd.set_option('display.max_columns', None)
    
    train.head()
    
    
    # In[11]:
    
    
    test.head()
    
    
    # In[12]:
    
    
    train.dtypes
    
    
    # In[13]:
    
    
    test.dtypes
    
    
    # In[14]:
    
    
    test['new_customer_id']=test.index + 1
    test['DOB'] = pd.to_datetime(test['DOB'])
    
    train['customer_id']=train['customer_id'].astype('object')
    train['DOB'] = pd.to_datetime(train['DOB'])
    
    
    # In[15]:
    
    
    current_date = pd.to_datetime('2017-12-31')
    
    test['customer_age'] = current_date.year - test['DOB'].dt.year
    
    test['customer_age'] -= ((current_date.month < test['DOB'].dt.month) | 
                                      ((current_date.month == test['DOB'].dt.month) & 
                                       (current_date.day < test['DOB'].dt.day))).astype(int)
    
    
    test['customer_age'] = test['customer_age'].astype("Int64")
    
    
    test = test[(test['customer_age'] >= 14) & (test['customer_age'] <= 100)]
    
    
    test['customer_age'].describe()
    
    
    # In[16]:
    
    
    test["tenure_valuation_mul"] = test["tenure"] * test["property_valuation"]
    test["tenure_valuation_div"] = test["tenure"] / test["property_valuation"]
    
    
    # In[17]:
    
    
    age_bins = [14, 18, 25, 35, 50, 65, 100]
    age_labels = ['Teen', 'Young Adult', 'Adult', 'Mid-age', 'Senior', 'Elderly']
    
    test['age_group'] = pd.cut(
        test['customer_age'],  
        bins=age_bins,
        labels=age_labels,
        right=True 
    )
    
    test[['customer_age', 'age_group']].head()
    
    
    # 
    # 
    # | **Nhóm tuổi**             | **Độ tuổi** | **Mô tả hành vi**                                                                 |
    # |---------------------------|-------------|------------------------------------------------------------------------------------|
    # | **Teen (14–18)**          | 14–18       | Học sinh, dùng xe đạp đi học, vận động nhẹ                                         |
    # | **Young Adult (19–25)**   | 19–25       | Sinh viên, người mới đi làm, thích khám phá                                       |
    # | **Adult (26–35)**         | 26–35       | Đạp xe thể thao, giữ dáng, lifestyle năng động                                    |
    # | **Mid-age (36–50)**       | 36–50       | Gắn bó ổn định, thích đạp xe rèn luyện sức khỏe                                   |
    # | **Senior (51–65)**        | 51–65       | Quan tâm sức khỏe, đạp xe nhẹ nhàng, thường xuyên bảo trì                         |
    # | **Elderly (66–100)**      | 66–100      | Ít vận động, nếu đạp thì là xe nhẹ, thường sử dụng xe đạp điện hỗ trợ             |
    # 
    
    # In[18]:
    
    
    train['age_group'] = pd.cut(
        train['customer_age'],  
        bins=age_bins,
        labels=age_labels,
        right=True 
    )
    
    
    # In[19]:
    
    
    train
    
    
    # In[20]:
    
    
    test
    
    
    # In[21]:
    
    
    cols = [
        'gender',
        'past_3_years_bike_related_purchases',
        'job_title',
        'job_industry_category',
        'wealth_segment',
        'deceased_indicator',
        'owns_car',
        'tenure',
        'property_valuation',
        'state',
        'customer_age',
        'age_group',
        'tenure_valuation_mul',
        'tenure_valuation_div',
        'clusters',
    ]
    
    train_df = train[cols]
    
    
    # In[22]:
    
    
    train_df
    
    
    # In[67]:
    
    
    newcus_df = test[[
        'gender',
        'past_3_years_bike_related_purchases',
        'job_title',
        'job_industry_category',
        'wealth_segment',
        'deceased_indicator',
        'owns_car',
        'tenure',
        'property_valuation',
        'state',
        'customer_age',
        'age_group',
        'tenure_valuation_mul',
        'tenure_valuation_div',
    ]]
    
    
    # In[68]:
    
    
    newcus_df
    
    
    # In[25]:
    
    
    train_df.select_dtypes(exclude='object').columns
    
    
    # # 2. Data preprocessing
    
    # In[26]:
    
    
    X_df_train=train_df[['gender', 'past_3_years_bike_related_purchases', 'job_title',
           'job_industry_category', 'wealth_segment', 'deceased_indicator',
           'owns_car', 'tenure', 'property_valuation', 'state', 'customer_age',
           'age_group', 'tenure_valuation_mul', 'tenure_valuation_div']]
    
    
    # In[27]:
    
    
    X_df_train
    
    
    # In[28]:
    
    
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    
    numerical_features = ['past_3_years_bike_related_purchases', 'tenure', 'property_valuation',
           'customer_age', 'tenure_valuation_mul',
           'tenure_valuation_div']
    categorical_features = ['gender', 'job_title', 'job_industry_category', 'wealth_segment','age_group',
           'deceased_indicator', 'owns_car', 'state']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    
    # In[29]:
    
    
    X = X_df_train
    
    
    # In[30]:
    
    
    X
    
    
    # In[31]:
    
    
    y=train_df['clusters']
    
    
    # In[32]:
    
    
    y
    
    
    # In[33]:
    
    
    y_counts = y.value_counts()
    plt.figure(figsize=(16, 6))
    sns.barplot(y=y_counts.index, x=y_counts.values,color='#0A7075', orient='h')
    plt.show()
    
    
    # In[34]:
    
    
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import precision_score , recall_score, accuracy_score
    
    from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
    from sklearn.metrics import classification_report, confusion_matrix
    
    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    
    from sklearn.exceptions import ConvergenceWarning
    
    from sklearn.model_selection import train_test_split
    
    
    # In[35]:
    
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=123,test_size=0.20)
    
    
    # In[36]:
    
    
    X_train
    
    
    # In[37]:
    
    
    y_train
    
    
    # # 3. Data modeling
    
    # ## 3.1. Logistic Regression
    
    # In[38]:
    
    
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    
    # In[ ]:
    
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    
    numerical_features = [
        'past_3_years_bike_related_purchases', 'tenure', 'property_valuation',
        'customer_age', 'tenure_valuation_mul', 'tenure_valuation_div'
    ]
    
    categorical_features = [
        'gender', 'job_title', 'job_industry_category', 'wealth_segment',
        'age_group', 'deceased_indicator', 'owns_car', 'state'
    ]
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    
    # In[50]:
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    logreg = LogisticRegression(random_state=123)
    pipeline = make_pipeline(preprocessor, logreg)
    
    param_grid = {
        'logisticregression__penalty': ['l1', 'l2'],
        'logisticregression__C': [0.1, 1, 10],
        'logisticregression__solver': ['liblinear'],
        'logisticregression__max_iter': [100, 300, 500]
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best CV Accuracy:", round(grid_search.best_score_, 4))
    
    
    # In[81]:
    
    
    best_logreg = grid_search.best_estimator_
    y_train_pred_logreg = best_logreg.predict(X_train)
    y_test_pred_logreg = best_logreg.predict(X_test)
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(confusion_matrix(y_train, y_train_pred_logreg), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix (Train) - Logistic Regression")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    sns.heatmap(confusion_matrix(y_test, y_test_pred_logreg), annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title("Confusion Matrix (Test) - Logistic Regression")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    
    print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred_logreg), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, y_test_pred_logreg), 4))
    
    # Classification Report
    print("\nClassification Report (Train):")
    print(classification_report(y_train, y_train_pred_logreg))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred_logreg))
    
    
    # ## 3.2. Random Forest
    
    # In[61]:
    
    
    rf_model = RandomForestClassifier(random_state=123)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf_model)
    ])
    
    param_distributions = {
        'classifier__n_estimators': randint(100, 300),
        'classifier__max_depth': randint(3, 20),
        'classifier__min_samples_split': randint(2, 10),
        'classifier__min_samples_leaf': randint(1, 10),
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__criterion': ['gini', 'entropy']
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    
    random_search_rf = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=skf,
        verbose=1,
        n_jobs=-1,
        random_state=123
    )
    
    random_search_rf.fit(X_train, y_train)
    
    print("Best Parameters Found:")
    print(random_search_rf.best_params_)
    print("Best Cross-Validation Score (during search):", round(random_search_rf.best_score_, 4))
    
    
    
    # In[62]:
    
    
    best_rf = random_search_rf.best_estimator_
    y_train_pred_rf = best_rf.predict(X_train)
    y_test_pred_rf = best_rf.predict(X_test)
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(confusion_matrix(y_train, y_train_pred_rf),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix (Train) - Random Forest")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    sns.heatmap(confusion_matrix(y_test, y_test_pred_rf),
                annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title("Confusion Matrix (Test) - Random Forest")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    
    acc_train = accuracy_score(y_train, y_train_pred_rf)
    acc_test = accuracy_score(y_test, y_test_pred_rf)
    
    print("Accuracy (Train):", round(acc_train, 4))
    print("Accuracy (Test):", round(acc_test, 4))
    
    # Classification Report 
    print("\nClassification Report (Train):")
    print(classification_report(y_train, y_train_pred_rf))
    
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred_rf))
    
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=skf, scoring='accuracy')
    print("Cross-Validation Accuracy (Train):", round(cv_scores.mean(), 4))
    
    
    # ## 3.3. HistGradientBoostingClassifier
    
    # In[64]:
    
    
    hgb_model = HistGradientBoostingClassifier(random_state=123)
    pipeline_hgb = make_pipeline(preprocessor, hgb_model)
    
    param_distributions_hgb = {
        'histgradientboostingclassifier__learning_rate': loguniform(0.001, 1),
        'histgradientboostingclassifier__l2_regularization': loguniform(1e-6, 1),
        'histgradientboostingclassifier__max_leaf_nodes': randint(10, 150),
        'histgradientboostingclassifier__min_samples_leaf': randint(10, 100),
        'histgradientboostingclassifier__max_bins': randint(4, 255),
    }
    
    random_search_hgb = RandomizedSearchCV(
        estimator=pipeline_hgb,
        param_distributions=param_distributions_hgb,
        n_iter=20,
        cv=skf,
        scoring='accuracy',
        verbose=1,
        random_state=123,
        n_jobs=-1
    )
    
    random_search_hgb.fit(X_train, y_train)
    
    print("Best Hyperparameters (HistGB):", random_search_hgb.best_params_)
    print("Best CV Accuracy (HistGB):", round(random_search_hgb.best_score_, 4))
    
    
    # In[65]:
    
    
    best_hgb = random_search_hgb.best_estimator_
    y_train_pred_hgb = best_hgb.predict(X_train)
    y_test_pred_hgb = best_hgb.predict(X_test)
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(confusion_matrix(y_train, y_train_pred_hgb),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix (Train) - HistGB")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    sns.heatmap(confusion_matrix(y_test, y_test_pred_hgb),
                annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title("Confusion Matrix (Test) - HistGB")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    
    print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred_hgb), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, y_test_pred_hgb), 4))
    
    # Classification Report
    print("\nClassification Report (Train):")
    print(classification_report(y_train, y_train_pred_hgb))
    
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred_hgb))
    
    
    # ## 3.4. XGB Classifer
    
    # In[51]:
    
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123)
    pipeline_xgb = make_pipeline(preprocessor, xgb_model)
    
    param_distributions_xgb = {
        'xgbclassifier__n_estimators': randint(100, 300),
        'xgbclassifier__max_depth': randint(3, 10),
        'xgbclassifier__learning_rate': uniform(0.01, 0.3),
        'xgbclassifier__subsample': uniform(0.5, 0.5),
        'xgbclassifier__colsample_bytree': uniform(0.5, 0.5),
        'xgbclassifier__gamma': uniform(0, 5),
        'xgbclassifier__reg_alpha': uniform(0, 1),
        'xgbclassifier__reg_lambda': uniform(0, 1)
    }
    
    random_search_xgb = RandomizedSearchCV(
        estimator=pipeline_xgb,
        param_distributions=param_distributions_xgb,
        n_iter=20,
        cv=skf,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=123
    )
    
    random_search_xgb.fit(X_train, y_train)
    
    print("Best Hyperparameters (XGBoost):", random_search_xgb.best_params_)
    print("Best CV Accuracy (XGBoost):", round(random_search_xgb.best_score_, 4))
    
    
    # In[87]:
    
    
    best_xgb = random_search_xgb.best_estimator_
    y_train_pred_xgb = best_xgb.predict(X_train)
    y_test_pred_xgb = best_xgb.predict(X_test)
    
    # Confusion Matrix 
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(confusion_matrix(y_train, y_train_pred_xgb),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix (Train) - XGBoost")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    sns.heatmap(confusion_matrix(y_test, y_test_pred_xgb),
                annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title("Confusion Matrix (Test) - XGBoost")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    
    print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred_xgb), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, y_test_pred_xgb), 4))
    
    # --- Classification Report ---
    print("\nClassification Report (Train):")
    print(classification_report(y_train, y_train_pred_xgb))
    
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred_xgb))
    
    
    # ## 3.5. MLP
    
    # In[70]:
    
    
    mlp_model = MLPClassifier(random_state=123, max_iter=500)
    pipeline_mlp = make_pipeline(preprocessor, mlp_model)
    
    param_grid_mlp = {
        'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'mlpclassifier__activation': ['relu', 'tanh'],
        'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
        'mlpclassifier__solver': ['adam', 'lbfgs']
    }
    
    grid_search_mlp = GridSearchCV(
        estimator=pipeline_mlp,
        param_grid=param_grid_mlp,
        cv=skf,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search_mlp.fit(X_train, y_train)
    
    print("Best Hyperparameters (MLP):", grid_search_mlp.best_params_)
    print("Best CV Accuracy (MLP):", round(grid_search_mlp.best_score_, 4))
    
    
    # In[71]:
    
    
    best_mlp = grid_search_mlp.best_estimator_
    y_train_pred_mlp = best_mlp.predict(X_train)
    y_test_pred_mlp = best_mlp.predict(X_test)
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(confusion_matrix(y_train, y_train_pred_mlp),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix (Train) - MLP")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    sns.heatmap(confusion_matrix(y_test, y_test_pred_mlp),
                annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title("Confusion Matrix (Test) - MLP")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    
    print("Train Accuracy:", round(accuracy_score(y_train, y_train_pred_mlp), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, y_test_pred_mlp), 4))
    
    # Classification Report 
    print("\nClassification Report (Train):")
    print(classification_report(y_train, y_train_pred_mlp))
    
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred_mlp))
    
    
    # # 4. Results
    
    # In[85]:
    
    
    summary = []
    
    # Random Forest
    summary.append({
        "Model": "Random Forest",
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred_rf), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_test_pred_rf), 4),
        "CV Score": round(random_search_rf.best_score_, 4)
    })
    
    # HistGradientBoosting
    summary.append({
        "Model": "HistGradientBoosting",
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred_hgb), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_test_pred_hgb), 4),
        "CV Score": round(random_search_hgb.best_score_, 4)
    })
    
    # XGBoost
    summary.append({
        "Model": "XGBoost",
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred_xgb), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_test_pred_xgb), 4),
        "CV Score": round(random_search_xgb.best_score_, 4)
    })
    
    # MLPClassifier
    summary.append({
        "Model": "MLPClassifier",
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred_mlp), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_test_pred_mlp), 4),
        "CV Score": round(grid_search_mlp.best_score_, 4)
    })
    
    # Logistic Regression
    summary.append({
        "Model": "Logistic Regression",
        "Train Accuracy": round(accuracy_score(y_train, y_train_pred_logreg), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_test_pred_logreg), 4),
        "CV Score": round(grid_search.best_score_, 4) 
    })
    
    accuracy_df = pd.DataFrame(summary)
    
    
    # In[89]:
    
    
    accuracy_df
    
    
    # In[ ]:
    
    
    
    
    
    # In[90]:
    
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    num_classes=4
    
    y_train_bin = label_binarize(y_train, classes=range(num_classes))
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    
    def plot_roc_curve_for_multiclass(fpr, tpr, auc_score, label, ax):
        ax.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.4f})')
    
    def plot_roc(model, model_name, X_train, y_train_bin, ax):
        y_train_prob = model.predict_proba(X_train)
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_train_prob[:, i])
            auc_score = auc(fpr, tpr)
            plot_roc_curve_for_multiclass(fpr, tpr, auc_score, f'{model_name} Class {i}', ax)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)') 
        ax.set_title(f'ROC Curve - {model_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    # --- Logistic Regression ---
    plot_roc(grid_search.best_estimator_, "Logistic Regression", X_train, y_train_bin, axs[0])
    
    # --- Random Forest ---
    plot_roc(random_search_rf.best_estimator_, "Random Forest", X_train, y_train_bin, axs[1])
    
    # --- HistGradientBoosting ---
    plot_roc(random_search_hgb.best_estimator_, "HistGradientBoosting", X_train, y_train_bin, axs[2])
    
    # --- XGBoost ---
    plot_roc(random_search_xgb.best_estimator_, "XGBoost", X_train, y_train_bin, axs[3])
    
    # --- MLPClassifier ---
    plot_roc(grid_search_mlp.best_estimator_, "MLPClassifier", X_train, y_train_bin, axs[4])
    
    # Hiển thị các biểu đồ
    plt.tight_layout()
    plt.show()
    
    
    # In[69]:
    
    
    best_xgb = random_search_xgb.best_estimator_  
    y_pred_newcus_xgb = best_xgb.predict(newcus_df)
    
    newcus_df['clusters_predict'] = y_pred_newcus_xgb
    
    
    # In[70]:
    
    
    newcus_df
    
    
    # In[93]:
    
    
    xgb_model = best_xgb.named_steps['xgbclassifier']
    
    feature_importances = xgb_model.feature_importances_
    
    if hasattr(preprocessor, 'transformers_'):
        feature_names = numerical_features + categorical_features  
    else:
        feature_names = X_train.columns  
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df)
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=pal[3])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance - XGBoost', fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()  
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', 
                 va='center', ha='left', color='black')
    
    plt.gca().invert_yaxis() 
    plt.show()
    
    
    # # Explore Newcus Cluster
    
    # In[90]:
    
    
    newcus_demo= newcus_df.select_dtypes(include=['Int64', 'Float64']).groupby('clusters_predict').mean().round(2)
    newcus_demo
    
    
    # In[78]:
    
    
    newcus_df.columns
    
    
    # In[81]:
    
    
    newcuscol=['gender', 'past_3_years_bike_related_purchases', 'job_title',
           'job_industry_category', 'wealth_segment', 'deceased_indicator',
           'owns_car', 'tenure', 'property_valuation', 'state', 'customer_age',
           'age_group', 'tenure_valuation_mul', 'tenure_valuation_div','clusters']
    
    
    # In[89]:
    
    
    oldcus_demo=train[newcuscol].select_dtypes(include=['Int64', 'Float64']).groupby('clusters').mean().round(2)
    oldcus_demo
    
    
    # In[131]:
    
    
    newcus_df['Customer_Type'] = 'New Customer'
    train['Customer_Type'] = 'Old Customer'
    combined_df = pd.concat([newcus_df, train])
    
    columns_to_compare = ['gender', 'past_3_years_bike_related_purchases', 'job_title', 
                          'job_industry_category', 'wealth_segment',
                          'owns_car', 'tenure', 'property_valuation', 'state', 
                          'customer_age', 'age_group', 'tenure_valuation_mul', 
                          'tenure_valuation_div', 'clusters']
    
    top_5_job_titles = combined_df['job_title'].value_counts().nlargest(5).index
    combined_df['job_title'] = combined_df['job_title'].where(combined_df['job_title'].isin(top_5_job_titles), 'Other')
    
    categorical_columns = [col for col in columns_to_compare if combined_df[col].dtype == 'object']
    categorical_columns = categorical_columns[:7]
    
    numeric_columns = [col for col in columns_to_compare if combined_df[col].dtype != 'object']
    
    plt.figure(figsize=(18, 15))
    for i, column in enumerate(numeric_columns):
        plt.subplot(4, 4, i + 1)
        sns.histplot(combined_df[column], kde=True, color=pal_1[0], label='New Customers', bins=30, alpha=0.6)
        sns.histplot(combined_df[combined_df['Customer_Type'] == 'Old Customer'][column], kde=True, color=pal_1[6], label='Old Customers', bins=30, alpha=0.6)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    # In[135]:
    
    
    # Plotting proportion plots for categorical columns
    plt.figure(figsize=(18, 10))
    for i, column in enumerate(categorical_columns):
        plt.subplot(2, 3, i + 1)
    
        customer_type_proportions = combined_df.groupby([column, 'Customer_Type']).size().unstack().fillna(0)
        customer_type_proportions = customer_type_proportions.div(customer_type_proportions.sum(axis=1), axis=0)
    
        customer_type_proportions.plot(kind='bar', stacked=True, ax=plt.gca(), alpha=0.7, color=[pal_1[0], pal_1[6]])
    
        plt.title(f'Proportion of {column} by Customer Type')
        plt.xlabel(column)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    
    # In[ ]:
    
    
    
    


if __name__ == '__main__':
    data_model_main()