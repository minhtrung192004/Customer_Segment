def customer_clustering_main():
    
    # In[1]:
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from matplotlib.colors import ListedColormap
    from matplotlib.colors import LinearSegmentedColormap
    
    from scipy import stats
    from scipy.stats import skew
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    from sklearn.impute import KNNImputer
    from sklearn.model_selection import RandomizedSearchCV
    from datetime import datetime as dt
    
    
    from scipy.stats.mstats import normaltest
    
    import warnings
    warnings.filterwarnings('ignore')
    
    
    # In[2]:
    
    
    #Define palette
    pal = ["#274D60","#6BA3BE","#0C969C","#0A7075","#032F30", "#031716"]
    cmap = ListedColormap(pal)
    palette = pal
    
    from matplotlib.colors import LinearSegmentedColormap
    pal_1 = ["#032F30","#0A7075","#0C969C","#fff9e6","#ffebc6","#fcd997","#f5b971"]
    palette_1 = pal_1
    
    reversed_pal_1 = pal_1[::-1]
    
    cmap_1 = LinearSegmentedColormap.from_list("reversed_blue_green_cmap", reversed_pal_1)
    
    
    # In[3]:
    
    
    plt.rcParams['font.family'] = 'Times New Roman'
    
    
    # In[4]:
    
    
    transaction_df=pd.read_csv('/Users/hominhtrung/Documents/Giáo Trình - FTU2/Năm 3/Phân tích dữ liệu/[PTDL] Final Project/transaction.csv', index_col=0)
    customerdemographic_df=pd.read_csv('/Users/hominhtrung/Documents/Giáo Trình - FTU2/Năm 3/Phân tích dữ liệu/[PTDL] Final Project/customerdemographic.csv', index_col=0)
    customeraddress_df=pd.read_csv('/Users/hominhtrung/Documents/Giáo Trình - FTU2/Năm 3/Phân tích dữ liệu/[PTDL] Final Project/customeraddress.csv', index_col=0)
    
    
    # # 1. Data Processing
    
    # In[5]:
    
    
    customerdemographic_df.shape
    
    
    # In[6]:
    
    
    customeraddress_df.shape
    
    
    # In[7]:
    
    
    transaction_df.head()
    
    
    # In[8]:
    
    
    customerdemographic_df.head()
    
    
    # In[9]:
    
    
    customerdemographic_df['tenure']
    
    
    # In[10]:
    
    
    customeraddress_df.head()
    
    
    # In[11]:
    
    
    customerdata = pd.merge(customerdemographic_df, customeraddress_df, how='inner', left_on='customer_id', right_on='customer_id')
    fact_order_detail = pd.merge(transaction_df, customerdata, how='inner', left_on='customer_id', right_on='customer_id')
    
    
    # In[12]:
    
    
    fact_order_detail
    
    
    # In[13]:
    
    
    fact_order_detail.info()
    
    
    # In[14]:
    
    
    fact_order_detail['transaction_id']=fact_order_detail['transaction_id'].astype('object')
    fact_order_detail['product_id']=fact_order_detail['product_id'].astype('object')
    fact_order_detail['customer_id']=fact_order_detail['customer_id'].astype('object')
    fact_order_detail['transaction_date'] = pd.to_datetime(fact_order_detail['transaction_date'])
    fact_order_detail['product_first_sold_date'] = pd.to_datetime(fact_order_detail['product_first_sold_date'])
    fact_order_detail['DOB'] = pd.to_datetime(fact_order_detail['DOB'])
    
    
    # In[15]:
    
    
    print(fact_order_detail.info())
    
    
    # In[16]:
    
    
    fact_order_detail.to_csv('fact_order_detail.csv')
    
    
    # In[17]:
    
    
    fact_order_detail['profit'] = fact_order_detail['list_price'] - fact_order_detail['standard_cost']
    
    
    # In[18]:
    
    
    start_date = pd.to_datetime("2017-01-01")
    end_date = pd.to_datetime("2017-12-31")
    today_date=pd.to_datetime("2017-12-31")
    
    
    # In[19]:
    
    
    fact_order_detail["customer_age"] = (end_date - fact_order_detail["DOB"]).dt.days // 365
    fact_order_detail["customer_age"] = fact_order_detail["customer_age"].astype("Int64")
    
    fact_order_detail = fact_order_detail[(fact_order_detail["customer_age"] >= 14) & (fact_order_detail["customer_age"] <= 100)]
    
    fact_order_detail.customer_age.describe()
    
    
    # In[20]:
    
    
    customerdata['customer_id']=customerdata['customer_id'].astype('object')
    customerdata['DOB'] = pd.to_datetime(customerdata['DOB'])
    
    
    # In[21]:
    
    
    current_date = pd.to_datetime('2017-12-31')
    
    customerdata['customer_age'] = current_date.year - customerdata['DOB'].dt.year
    
    customerdata['customer_age'] -= ((current_date.month < customerdata['DOB'].dt.month) | 
                                      ((current_date.month == customerdata['DOB'].dt.month) & 
                                       (current_date.day < customerdata['DOB'].dt.day))).astype(int)
    
    
    customerdata['customer_age'] = customerdata['customer_age'].astype("Int64")
    
    
    customerdata = customerdata[(customerdata['customer_age'] >= 14) & (customerdata['customer_age'] <= 100)]
    
    
    customerdata['customer_age'].describe()
    
    
    # In[22]:
    
    
    fact_order_detail["product_first_sold_date"] = pd.to_datetime(fact_order_detail["product_first_sold_date"], errors='coerce')
    end_date = pd.to_datetime('2017-12-31')
    
    fact_order_detail["product_age"] = (end_date - fact_order_detail["product_first_sold_date"]).dt.days // 365
    
    fact_order_detail = fact_order_detail.dropna(subset=["product_age"])
    
    fact_order_detail["product_age"].describe()
    
    
    # In[23]:
    
    
    fact_order_detail["profit"] = fact_order_detail["list_price"] - fact_order_detail["standard_cost"]
    
    # Calculate recency
    fact_order_detail["recency"] = (today_date - fact_order_detail["transaction_date"]).dt.days.astype("Int64")
    
    # Calculate frequency
    
    fact_order_detail["frequency"] = fact_order_detail.groupby("customer_id")["customer_id"].transform( "count")
    
    
    # In[24]:
    
    
    fact_order_detail
    
    
    # In[25]:
    
    
    fact_order_detail.info()
    
    
    # 
    
    # In[26]:
    
    
    fact_order_detail.groupby('brand').agg({'profit': lambda x: x.sum()})
    
    
    # In[27]:
    
    
    pivot_table = fact_order_detail.pivot_table(index='customer_id', columns='brand', values='profit', aggfunc='sum', fill_value=0)
    pivot_table
    
    
    # In[28]:
    
    
    # Generating the RFM Table
    RfmTable = fact_order_detail.groupby("customer_id").agg(
        {
            "recency": lambda x: x.min(),
            "frequency": lambda x: x.count(),
            "profit": lambda x: x.sum(),
        }
    )
    
    RfmTable.rename(
        columns={
            "recency": "recency",
            "frequency": "frequency",
            "profit": "monetary",
        },
        inplace=True,
    )
    
    RfmTable.head()
    
    
    # In[29]:
    
    
    customerdata_=customerdata.merge(RfmTable, how = 'inner', on='customer_id')
    customerdata_merge=customerdata_.merge(pivot_table, how = 'inner', on='customer_id')
    
    
    # In[30]:
    
    
    customerdata_merge
    
    
    # In[31]:
    
    
    customerdata_merge["tenure_valuation_mul"] = customerdata_merge["tenure"] * customerdata_merge["property_valuation"]
    customerdata_merge["tenure_valuation_div"] = customerdata_merge["tenure"] / customerdata_merge["property_valuation"]
    
    
    # In[32]:
    
    
    customerdata_merge.info()
    
    
    # # 2. PCA - Features Selection
    
    # In[33]:
    
    
    customerdata_merge_df=customerdata_merge.copy()
    
    
    # In[34]:
    
    
    # Hadeling Outlier
    def replace_outliers_with_zscore(data, column):
        col_values = data[column]
        mean = np.mean(col_values)
        std = np.std(col_values)
    
        # Manual Z-score calculation
        z_scores = (col_values - mean) / std
        abs_z_scores = np.abs(z_scores)
    
        median = np.median(col_values)
    
        data[column] = np.where(abs_z_scores > 3, median, col_values)
    
    numerical_columns = customerdata_merge_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        replace_outliers_with_zscore(customerdata_merge_df, col)
    
    
    # In[35]:
    
    
    numerical_columns = customerdata_merge_df.select_dtypes(include=['float64', 'int64']).columns
    
    num_cols = len(numerical_columns)
    cols = 3  
    rows = (num_cols + cols - 1) // cols  
    
    plt.figure(figsize=(16, 2 * rows))
    
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x=customerdata_merge_df[col], color='#0A7075', orient='h')
        plt.title(f'Boxplot of {col}', fontsize=12)
        plt.tight_layout()
    
    plt.suptitle('Boxplots of Numerical Columns After Outlier Replacement', fontsize=16, y=1.02)
    plt.show()
    
    
    # In[36]:
    
    
    plt.figure(figsize=(16, 3.5 * rows))
    
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(rows, cols, i)    
        sns.distplot(customerdata_merge_df[col], bins=30, kde=True, color='#0A7075')
        plt.title(f'Histogram of {col}', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.suptitle('Histograms of Numerical Columns After Outlier Replacement', fontsize=16,  fontweight = 'bold', y=1)
    plt.tight_layout()
    plt.show()
    
    
    # In[37]:
    
    
    data_rfm = customerdata_merge_df.copy()
    
    
    # In[38]:
    
    
    data_rfm.info()
    
    
    # ## 2.1. Data Pre-Processing
    
    # In[39]:
    
    
    data_rfm.drop(['first_name','last_name', 'DOB', 'deceased_indicator', 'address','postcode','country', 'job_title'], axis=1, inplace=True)
    
    
    # In[40]:
    
    
    numerical_columns=data_rfm.select_dtypes(include=('int64','float64'))
    
    
    # In[41]:
    
    
    plt.figure(figsize=(16, 3.5 * rows))
    
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(rows, cols, i)    
        sns.distplot(data_rfm[col], bins=30, kde=True, color='#6BA3BE')
        plt.title(f'Histogram of {col}', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.suptitle('Histograms of Numerical Columns After Outlier Replacement', fontsize=16,  fontweight = 'bold', y=1)
    plt.tight_layout()
    plt.show()
    
    
    # In[42]:
    
    
    def calculate_skewness(df):
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        skew_vals = df[num_cols].apply(lambda x: skew(x.dropna()))
    
        skew_df = pd.DataFrame({
            'Variable': skew_vals.index,
            'Skewness': skew_vals.values
        }).sort_values(by='Skewness', key=abs, ascending=False).reset_index(drop=True)
    
        return skew_df
    calculate_skewness(data_rfm)
    
    
    # In[43]:
    
    
    def best_skew_fix(data, threshold=0.5):
        result = []
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
        for col in num_cols:
            original = data[col]
            orig_skew = skew(original.dropna())
    
            if abs(orig_skew) > threshold:
                transformations = {}
    
                # Log1p
                try:
                    log_skew = skew(np.log1p(original[original >= 0]))
                    transformations['log1p'] = log_skew
                except:
                    transformations['log1p'] = np.nan
    
                # Sqrt
                try:
                    sqrt_skew = skew(np.sqrt(original[original >= 0]))
                    transformations['sqrt'] = sqrt_skew
                except:
                    transformations['sqrt'] = np.nan
    
                # Box-Cox ---
                try:
                    boxcox_skew = skew(boxcox(original[original > 0])[0])
                    transformations['boxcox'] = boxcox_skew
                except:
                    transformations['boxcox'] = np.nan
    
                best_method = min(transformations, key=lambda k: abs(transformations[k]) if pd.notnull(transformations[k]) else np.inf)
                best_skew = transformations[best_method]
    
                result.append({
                    'Column': col,
                    'Original_Skew': orig_skew,
                    'Best_Method': best_method,
                    'Transformed_Skew': best_skew
                })
    
        return pd.DataFrame(result)
    
    
    # In[44]:
    
    
    best_skew_fix(data_rfm, threshold=0.5)
    
    
    # In[45]:
    
    
    from scipy.stats import boxcox
    skew_fixes = best_skew_fix(data_rfm)
    
    data_rfm_transformed = data_rfm.copy()
    
    for _, row in skew_fixes.iterrows():
        col = row['Column']
        method = row['Best_Method']
    
        if method == 'log1p':
            data_rfm_transformed[col] = np.log1p(data_rfm_transformed[col])
        elif method == 'sqrt':
            data_rfm_transformed[col] = np.sqrt(data_rfm_transformed[col])
        elif method == 'boxcox':
            positive_data = data_rfm_transformed[col][data_rfm_transformed[col] > 0]
            if not positive_data.empty:
                transformed, _ = boxcox(positive_data)
                data_rfm_transformed.loc[positive_data.index, col] = transformed
    
    
    # In[46]:
    
    
    new_skew_df = calculate_skewness(data_rfm_transformed)
    new_skew_df
    
    
    # In[47]:
    
    
    data_rfm_transformed
    
    
    # In[48]:
    
    
    plt.figure(figsize=(16, 3.5 * rows))
    
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(rows, cols, i)    
        sns.distplot(data_rfm_transformed[col], bins=30, kde=True, color='#6BA3BE')
        plt.title(f'Histogram of {col}', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.suptitle('Histograms of Numerical Columns After Outlier Replacement', fontsize=16, fontweight = 'bold', y=1)
    plt.tight_layout()
    plt.show()
    
    
    # In[49]:
    
    
    data_rfm=data_rfm_transformed.copy()
    
    
    # In[50]:
    
    
    data_rfm.info()
    
    
    # In[51]:
    
    
    data_rfm.select_dtypes(include='object').columns
    
    
    # In[52]:
    
    
    category_cols=['gender', 'job_industry_category',
           'wealth_segment', 'owns_car', 'state']
    
    
    # In[53]:
    
    
    # Convert all object data type columns to categorical features
    for col in category_cols:
        data_rfm[col] = data_rfm[col].astype('category')
    
    
    # In[54]:
    
    
    data_rfm.info()
    
    
    # In[55]:
    
    
    data_rfm['customer_id']=data_rfm['customer_id'].astype('int64')
    
    
    # In[56]:
    
    
    for col in category_cols:
        print(data_rfm[col].value_counts())
    
    
    # In[57]:
    
    
    data_rfm['job_industry_category'].value_counts()
    
    
    # In[58]:
    
    
    from sklearn.preprocessing import LabelEncoder
    
    
    # In[59]:
    
    
    label_encoder = LabelEncoder()
    
    for col in data_rfm.select_dtypes(include=['category']).columns:
        data_rfm[col] = label_encoder.fit_transform(data_rfm[col])
    for col in data_rfm.select_dtypes(include = ['bool']).columns:
        data_rfm[col] = data_rfm[col].astype('int')
    
    data_rfm.head()
    
    
    # In[60]:
    
    
    data_rfm
    
    
    # In[ ]:
    
    
    
    
    
    # In[61]:
    
    
    data_rfm.info()
    
    
    # In[62]:
    
    
    ds=data_rfm.select_dtypes(include=['int64', 'float64'])
    
    
    # In[63]:
    
    
    ds.drop('customer_id', inplace=True, axis=1)
    
    
    # In[64]:
    
    
    ds
    
    
    # In[65]:
    
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler() 
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns)
    
    
    # In[66]:
    
    
    scaled_ds
    
    
    # In[67]:
    
    
    scaled_ds_columns=scaled_ds.columns
    
    
    # In[68]:
    
    
    scaled_ds_columns
    
    
    # In[69]:
    
    
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    # ## principal conponent analysis
    
    # In[70]:
    
    
    optimal_components = 18
    pca = PCA(n_components = optimal_components)
    data_pca = pca.fit_transform(scaled_ds)
    
    feature_names = scaled_ds_columns
    pca_df = pd.DataFrame(pca.components_, columns=feature_names)
    
    explained_variance = pca.explained_variance_ratio_
    
    pca_df_with_variance = pd.DataFrame(pca.components_, columns=feature_names)
    pca_df_with_variance['Explained Variance'] = explained_variance
    
    plt.figure(figsize=(20, 8))
    sns.heatmap(pca_df_with_variance.T, annot=True, cmap=cmap_1, fmt='.2f', center=0, linecolor='white', linewidths=0.7)
    plt.title("Contribution of Customers' Features to Principal Components", fontsize=16, fontweight='bold', y=1.02)
    plt.xlabel('Principal Components')
    plt.ylabel("Original Customers' Features")
    plt.show()
    
    
    # # 3. Customer Segment - Clustering - KMenas Method
    
    # #
    
    # In[124]:
    
    
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    def evaluate_clustering(X, clustering_algo):
        clustering_algo.fit(X)
        labels = clustering_algo.labels_
        if len(set(labels)) == 1:
            return None, None, None  
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        return silhouette, davies_bouldin, calinski_harabasz
    
    # Evaluate KMeans
    kmeans = KMeans(n_clusters=4, random_state=123)
    kmeans_metrics = evaluate_clustering(data_pca, kmeans)
    
    # Evaluate Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=4)
    hierarchical_metrics = evaluate_clustering(data_pca, hierarchical)
    
    # Evaluate DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(data_pca)
    labels_dbscan = dbscan.labels_
    
    if len(set(labels_dbscan)) > 1:
        dbscan_metrics = (
            silhouette_score(data_pca, labels_dbscan),
            davies_bouldin_score(data_pca, labels_dbscan),
            calinski_harabasz_score(data_pca, labels_dbscan)
        )
    else:
        dbscan_metrics = ('Not applicable (only one cluster found)', ) * 3
    
    
    results = {
        'Clustering Method': ['KMeans', 'Hierarchical', 'DBSCAN'],
        'Silhouette Score': [kmeans_metrics[0], hierarchical_metrics[0], dbscan_metrics[0]],
        'Davies-Bouldin Index': [kmeans_metrics[1], hierarchical_metrics[1], dbscan_metrics[1]],
        'Calinski-Harabasz Index': [kmeans_metrics[2], hierarchical_metrics[2], dbscan_metrics[2]],
    }
    
    results_df = pd.DataFrame(results)
    results_df
    
    
    # Dựa trên các chỉ số đánh giá, KMeans là phương pháp tốt nhất trong ba lựa chọn, dù hiệu quả vẫn chưa cao (Silhouette Score thấp và Davies-Bouldin Index khá lớn). Hierarchical clustering cho kết quả kém hơn với các chỉ số thấp hơn. DBSCAN không hiệu quả vì chỉ tìm được một cụm, có thể do tham số chưa phù hợp.
    # 👉 Gợi ý: Nên tiếp tục dùng KMeans và tối ưu thêm, hoặc điều chỉnh tham số để cải thiện DBSCAN
    
    # In[125]:
    
    
    scaled_ds.columns
    
    
    # In[126]:
    
    
    selected_cols=['property_valuation','frequency','recency','tenure','customer_age',
           'monetary', 'Giant Bicycles', 'Norco Bicycles', 'OHM Cycles', 'Solex',
           'Trek Bicycles', 'WeareA2B','tenure_valuation_mul','tenure_valuation_div']
    
    
    # In[127]:
    
    
    scaled_ds[selected_cols]
    
    
    # In[140]:
    
    
    PCA_ds = scaled_ds[selected_cols]
    
    
    # In[141]:
    
    
    PCA_ds
    
    
    # In[142]:
    
    
    # ELBOW METHOD
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    def calculate_inertia(data, max_clusters):
        inertia = []
        for n in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=123)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)
        return inertia
    max_clusters = 20
    inertia = calculate_inertia(PCA_ds, max_clusters)
    
    plt.figure(figsize=(16, 6))
    plt.plot(range(1, max_clusters + 1), inertia, marker = 'o', color='#f5b971')
    plt.title('Elbow Method for Optimal Number of Clusters',  fontweight = 'bold', fontsize=16, y=1)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.show()
    
    
    # In[143]:
    
    
    def kmeans(normalised_df_rfm, clusters_number, original_df_rfm):
    
        kmeans = KMeans(n_clusters = clusters_number, random_state = 123)
        kmeans.fit(normalised_df_rfm)
        cluster_labels = kmeans.labels_  
        df_new = original_df_rfm.assign(Cluster = cluster_labels)
    
        model = TSNE(random_state=123)
        transformed = model.fit_transform(df_new)
    
        plt.title('Flattened Graph of {} Clusters'.format(clusters_number),  fontweight = 'bold', fontsize=14, y=1)
        sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette='pastel')
    
        return df_new
    
    
    # In[144]:
    
    
    plt.figure(figsize=(25, 25))
    plt.subplot(3, 1, 1)
    df_rfm_k3 = kmeans(PCA_ds, 3, PCA_ds)
    plt.subplot(3, 1, 2)
    df_rfm_k4 = kmeans(PCA_ds, 4, PCA_ds)
    plt.subplot(3, 1, 3)
    df_rfm_k5 = kmeans(PCA_ds, 5, PCA_ds)
    plt.tight_layout()
    plt.show()
    
    
    # In[145]:
    
    
    np.random.seed(123)
    KM = KMeans(n_clusters=4, random_state = 123)
    
    yhat_KM = KM.fit_predict(PCA_ds)
    PCA_ds["Clusters"] = yhat_KM
    
    
    # In[146]:
    
    
    PCA_ds
    
    
    # In[85]:
    
    
    customerdata_merge['clusters'] = KM.labels_
    
    
    # In[86]:
    
    
    customerdata_merge
    
    
    # In[87]:
    
    
    customerdata_merge['clusters'].value_counts()
    
    
    # In[88]:
    
    
    customerdata_merge
    
    
    # In[89]:
    
    
    customerdata_merge.dtypes
    
    
    # In[90]:
    
    
    mean_Kmeans = customerdata_merge.select_dtypes(include=['int64', 'float64', 'int32']).groupby('clusters').mean()
    mean_Kmeans
    
    
    # In[91]:
    
    
    ['past_3_years_bike_related_purchases','tenure','property_valuation','customer_age','recency','frequency','monetary','Giant Bicycles'
     'Norco Bicycles','OHM Cycles','OHM Cycles','Trek Bicycles','WeareA2B']
    
    
    # In[92]:
    
    
    ds_final=customerdata_merge[['past_3_years_bike_related_purchases','tenure','property_valuation','customer_age','recency','frequency','monetary','Giant Bicycles',
     'Norco Bicycles','OHM Cycles','OHM Cycles','Trek Bicycles','WeareA2B','clusters']]
    
    
    # In[93]:
    
    
    mean_Kmeans=ds_final.groupby('clusters').mean()
    mean_Kmeans
    
    
    # In[94]:
    
    
    mean_Kmeans.to_csv('final_demo_clusters.csv')
    
    
    # In[95]:
    
    
    customerdata_merge
    
    
    # In[96]:
    
    
    customerdata_merge['clusters'] = KM.labels_
    
    
    # In[97]:
    
    
    customerdata_merge.to_csv('customer_data_segmented.csv')
    
    
    # In[98]:
    
    
    customerdata_merge.select_dtypes(include=['int64', 'float64', 'int32']).groupby('clusters').mean()
    
    
    # In[106]:
    
    
    customerdata_merge
    
    
    # In[ ]:
    
    
    # # Multiplying and Dividing the tenure and property_valuation
    
    # customerdata_merge["tenure_valuation_mul"] = customerdata_merge["tenure"] * customerdata_merge["property_valuation"]
    # customerdata_merge["tenure_valuation_div"] = customerdata_merge["tenure"] / customerdata_merge["property_valuation"]
    
    
    # In[99]:
    
    
    demodescribe = customerdata_merge[['tenure','customer_age','recency','frequency','monetary','clusters','tenure_valuation_mul','tenure_valuation_div']].groupby('clusters').mean()
    
    
    # In[102]:
    
    
    demodescribe
    
    
    # In[103]:
    
    
    demodescribe_df = demodescribe.reset_index()
    
    
    # In[104]:
    
    
    demodescribe_df
    
    
    # In[105]:
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    
    # Scatter plot 1: Frequency vs Monetary
    sns.scatterplot(
        data=customerdata_merge,
        x='frequency',
        y='monetary',
        hue='clusters',
        palette='Set2',
        ax=axes[0],
        s=80  # kích thước điểm
    )
    axes[0].set_title('Frequency vs Monetary', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Monetary ($)')
    
    # Scatter plot 2: Recency vs Monetary
    sns.scatterplot(
        data=customerdata_merge,
        x='recency',
        y='monetary',
        hue='clusters',
        palette='Set2',
        ax=axes[1],
        s=80
    )
    axes[1].set_title('Recency vs Monetary', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Recency')
    axes[1].set_ylabel('Monetary ($)')
    
    # Canh chỉnh bố cục và legend
    plt.tight_layout()
    axes[1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[ ]:
    
    
    
    


if __name__ == '__main__':
    customer_clustering_main()