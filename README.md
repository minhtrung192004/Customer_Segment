# 📊 PHÂN KHÚC KHÁCH HÀNG  VÀ ĐỀ XUẤ CHIẾN LƯỢC KINH DOANH DỰA TRÊN RFM, PCA VÀ PHÂN TICH CORHORT

Dự án này phân tích dữ liệu giao dịch và nhân khẩu học của khách hàng từ một công ty bán lẻ xe đạp tại Úc, nhằm khám phá các nhóm khách hàng chính, theo dõi hành vi tiêu dùng và đề xuất các chiến lược marketing dựa trên dữ liệu.

---

## 📁 Tổng Quan Dự Án

- **Mục tiêu**: Phân khúc khách hàng, phân tích hành vi và xây dựng chiến lược giữ chân, marketing cá nhân hóa.
- **Dữ liệu**: Giao dịch, thông tin nhân khẩu học, địa chỉ khách hàng và khách hàng mới (năm 2017).
- **Công cụ sử dụng**: Python, Power BI, PCA, KMeans, RFM, phân tích cohort.

---

## 📌 Cấu Trúc Dự Án

### 1. 🏢 Bối Cảnh

Hiểu được hành vi khách hàng là yếu tố then chốt giúp doanh nghiệp cạnh tranh và phát triển bền vững. Dự án áp dụng các kỹ thuật phân tích dữ liệu hiện đại (RFM, PCA, KMeans, Cohort Analysis) để khám phá các nhóm khách hàng, theo dõi tỷ lệ giữ chân và hỗ trợ quyết định chiến lược cho doanh nghiệp.

---

### 2. 🧹 Làm Sạch Dữ Liệu

**Các bộ dữ liệu được xử lý gồm:**

#### a. `Transactions`
- Không có dòng trùng lặp;
- Loại bỏ dòng thiếu giá trị (`brand`, `product_line`, `standard_cost`);
- Chuẩn hóa cột ngày (`transaction_date`) thành định dạng thời gian;
- Tính toán biên lợi nhuận (`profit_margin`).

#### b. `Customer Demographic`
- Thay thế giá trị thiếu trong `job_title`, `industry_category` bằng “Others”;
- Loại bỏ khách hàng có `DOB` bất hợp lý (năm 1843) và `gender = "U"`;
- Tính thêm biến `customer_age`, `customer_value_score`.

#### c. `Customer Address`
- Gộp với bảng demographic qua `customer_id`;
- Dùng `postcode` và `property_valuation` để phân vùng theo địa lý.

#### d. `New Customer List`
- Làm sạch tương tự bảng nhân khẩu học;
- Loại bỏ dòng chứa giá trị sai định dạng;
- Chuẩn hóa để sẵn sàng dự đoán phân khúc.

---

### 3. 👥 Phân Khúc Khách Hàng

- **RFM Scoring**: Tính Recency, Frequency và Monetary cho từng khách hàng.
- **PCA**: Giảm chiều dữ liệu trước khi phân cụm.
- **KMeans**: Phân cụm và xác định số lượng nhóm tối ưu (sử dụng Elbow Method).
  - Gồm 4 nhóm chính:
    - Khách hàng trung thành
    - Khách hàng giá trị cao
    - Khách hàng không còn hoạt động
    - Khách hàng mới/tiềm năng
- **TSNE**: Trực quan hóa kết quả phân cụm.

---

### 4. 📊 Phân Tích Dữ Liệu (EDA)

Phân tích từ tổng quan đến chi tiết theo từng nhóm:

- **Hiệu quả kinh doanh**
  - Doanh thu theo thời gian, khu vực, danh mục sản phẩm.
  - Lợi nhuận theo thương hiệu và nhóm hàng.
  - Tỷ lệ hủy đơn theo loại sản phẩm.

- **Hành vi khách hàng**
  - Phân tích nhân khẩu học: độ tuổi, giới tính, nghề nghiệp, tài sản.
  - Tần suất mua hàng, giá trị trung bình đơn hàng.
  - **Cohort Analysis**: phân tích tỷ lệ giữ chân theo tháng.

Trực quan hóa bằng:
- Biểu đồ cột, đường, heatmap, Sankey chart (Python + Power BI)

---

### 🧠 5. Mô hình hóa & Dự đoán khách hàng mới

#### 5.1. Mô hình dữ liệu và Pipeline
- Dữ liệu được chia thành tập huấn luyện (80%) và kiểm tra (20%).
- Sử dụng `ColumnTransformer` để chuẩn hóa và mã hóa đặc trưng (standardization cho biến số, One-Hot Encoding cho biến phân loại).
- Tối ưu hóa hyperparameter thông qua `GridSearchCV` (cho Logistic Regression, MLP) và `RandomizedSearchCV` (cho Random Forest, XGBoost, HistGradientBoosting).

---

#### 5.2. Các mô hình sử dụng
| Mô hình                     | Kỹ thuật tối ưu             | Cross-validation Score |
|-----------------------------|------------------------------|-------------------------|
| Logistic Regression         | GridSearchCV                 | 0.7563                  |
| Random Forest               | RandomizedSearchCV           | 0.9718                  |
| HistGradientBoosting        | RandomizedSearchCV           | 0.9820                  |
| XGBoost                     | RandomizedSearchCV           | 0.9586                  |
| MLP Classifier              | GridSearchCV                 | 0.9738                  |

---

#### 5.3. Kết quả huấn luyện các mô hình

| Mô hình                     | Accuracy (Train) | Accuracy (Test) | CV Score |
|-----------------------------|------------------|------------------|----------|
| Logistic Regression         | 75.64%           | 75.40%           | 75.63%   |
| Random Forest               | 99.40%           | 98.03%           | 97.18%   |
| HistGradientBoosting        | 99.92%           | 99.02%           | 98.20%   |
| XGBoost                     | 98.71%           | 96.33%           | 95.86%   |
| MLP Classifier              | 99.22%           | 97.93%           | 97.38%   |

🎯 **Nhận xét**:
- HistGradientBoosting và Random Forest đạt hiệu suất cao, không bị overfitting.
- XGBoost có dấu hiệu overfitting nhẹ.
- Logistic Regression cho kết quả thấp nhất, phù hợp với mô hình baseline.

---

#### 5.4. Hyperparameter Tối ưu

| Mô hình                     | Hyperparameter                          | Giá trị tối ưu                                   |
|-----------------------------|------------------------------------------|--------------------------------------------------|
| **Logistic Regression**     | `penalty`                                | `'l1'`                                           |
|                             | `C`                                      | `10`                                             |
|                             | `solver`                                 | `'liblinear'`                                    |
| **Random Forest**           | `n_estimators`                           | `204`                                            |
|                             | `criterion`                              | `'entropy'`                                      |
|                             | `max_depth`                              | `18`                                             |
|                             | `max_features`                           | `'log2'`                                         |
|                             | `min_samples_split`                      | `7`                                              |
|                             | `min_samples_leaf`                       | `1`                                              |
| **HistGradientBoosting**    | `learning_rate`                          | `0.6799`                                         |
|                             | `l2_regularization`                      | `0.2293`                                         |
|                             | `max_leaf_nodes`                         | `119`                                            |
|                             | `min_samples_leaf`                       | `36`                                             |
|                             | `max_bins`                               | `74`                                             |
| **XGBoost**                 | `colsample_bytree`                       | `0.9212`                                         |
|                             | `gamma`                                  | `0.416`                                          |
|                             | `learning_rate`                          | `0.2391`                                         |
|                             | `max_depth`                              | `4`                                              |
|                             | `n_estimators`                           | `298`                                            |
|                             | `reg_alpha`                              | `0.1942`                                         |
|                             | `reg_lambda`                             | `0.5725`                                         |
|                             | `subsample`                              | `0.5479`                                         |
| **MLP Classifier**          | `hidden_layer_sizes`                     | `(100, 50)`                                      |
|                             | `activation`                             | `'tanh'`                                         |
|                             | `alpha`                                  | `0.01`                                           |
|                             | `solver`                                 | `'adam'`                                         |

---

#### 5.5. Đánh giá mô hình bằng ROC & Classification Report

- **HistGradientBoosting** và **Random Forest** có AUC gần 1.000 cho tất cả các lớp → mô hình ổn định, tổng quát tốt.
- **XGBoost** đạt AUC cao ở lớp 0 và 1, giảm nhẹ ở lớp 2 và 3.
- **MLP Classifier** hoạt động tốt ở lớp phổ biến, nhưng có sự giảm nhẹ ở lớp ít dữ liệu.
- **Logistic Regression** có AUC thấp, đặc biệt là lớp 2 chỉ đạt 0.7452 → hiệu suất kém hơn hẳn.

---

#### 5.6. Dự đoán khách hàng mới

- Mô hình XGBoost dự đoán hiệu quả các phân khúc khách hàng mới dựa trên hành vi và đặc điểm nhân khẩu học.
- Nhóm khách hàng tiềm năng (high-value) có giá trị tài sản cao và thời gian gắn bó dài, cho thấy tính khả thi trong việc phát triển chiến lược chăm sóc cá nhân hóa.

---

✅ **Kết luận**:
- **HistGradientBoosting** là lựa chọn tốt nhất trong bài toán phân khúc khách hàng mới.
- Mô hình có độ chính xác cao, khả năng tổng quát tốt và tránh overfitting.

