# 📊 PHÂN KHÚC KHÁCH HÀNG  VÀ ĐỀ XUẤ CHIẾN LƯỢC KINH DOANH DỰA TRÊN RFM, PCA VÀ PHÂN TICH CORHORT
Dự án này phân tích dữ liệu giao dịch và nhân khẩu học của khách hàng từ một công ty bán lẻ xe đạp tại Úc, nhằm khám phá các nhóm khách hàng chính, theo dõi hành vi tiêu dùng và đề xuất các chiến lược marketing dựa trên dữ liệu.
## 📁 Tổng Quan Dự Án
- **Mục tiêu**: Phân khúc khách hàng, phân tích hành vi và xây dựng chiến lược giữ chân, marketing cá nhân hóa.
- **Dữ liệu**: Giao dịch, thông tin nhân khẩu học, địa chỉ khách hàng và khách hàng mới (năm 2017).
- **Công cụ sử dụng**: Python, Power BI, PCA, KMeans, RFM, phân tích cohort.
## 📌 Cấu Trúc Dự Án
### 1. 🏢 Bối Cảnh
Hiểu được hành vi khách hàng là yếu tố then chốt giúp doanh nghiệp cạnh tranh và phát triển bền vững. Dự án áp dụng các kỹ thuật phân tích dữ liệu hiện đại (RFM, PCA, KMeans, Cohort Analysis) để khám phá các nhóm khách hàng, theo dõi tỷ lệ giữ chân và hỗ trợ quyết định chiến lược cho doanh nghiệp.
### 2. Làm Sạch Dữ Liệu
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

### 3. Phân khúc Khách hàng (Customer Segmentation)

Sau quá trình xử lý dữ liệu và tính toán RFM (Recency – Frequency – Monetary), nhóm thực hiện:
- Chuẩn hóa dữ liệu RFM và các thuộc tính nhân khẩu học.
- Áp dụng **PCA** để giảm chiều dữ liệu nhằm tăng hiệu quả phân cụm.
- Sử dụng **KMeans** (với k = 4, xác định bằng Elbow Method và xác nhận bằng TSNE) để phân nhóm khách hàng.

Kết quả cho ra **4 nhóm khách hàng chính**, mỗi nhóm có những đặc điểm hành vi và nhân khẩu học riêng biệt như sau:

#### 🟦 Nhóm 0 – Khách hàng trung thành (Loyal Customers)

- **Recency**: Trung bình (≈ 55 ngày)  
- **Frequency**: Cao (≈ 6 đơn hàng)  
- **Monetary**: Trung bình  
- **Tenure**: Cao (≈ 11.7 năm)  
- **Đặc điểm khác**:
  - Phân bố khá đều giữa các vùng.
  - Thường ở độ tuổi trung niên (~35–45).
  - Giá trị tài sản ở mức trung bình-khá.
- **Chiến lược gợi ý**:
  - Duy trì lòng trung thành bằng các chương trình tích điểm, quà tặng tri ân, hoặc ưu đãi định kỳ.
  - Tăng tương tác thông qua email marketing, ưu đãi theo hành vi.
  - 
#### 🟩 Nhóm 1 – Khách hàng mới tiềm năng (New Customers)

- **Recency**: Thấp (giao dịch gần đây)
- **Frequency**: Thấp
- **Monetary**: Thấp
- **Tenure**: Rất thấp (≈ dưới 1 năm)
- **Đặc điểm khác**:
  - Chủ yếu là người trẻ (~20–30 tuổi).
  - Mới bắt đầu tương tác với doanh nghiệp.
  - Phân bố đều trên các kênh Online và Offline.
- **Chiến lược gợi ý**:
  - Tăng cường onboarding (hướng dẫn sử dụng sản phẩm/dịch vụ).
  - Cung cấp ưu đãi lần đầu và khuyến mãi “mua lại lần 2”.
  - Theo dõi để chuyển họ thành nhóm trung thành.

#### 🟥 Nhóm 2 – Khách hàng không còn hoạt động (Inactive Customers)

- **Recency**: Cao (lâu không mua hàng)
- **Frequency**: Rất thấp
- **Monetary**: Rất thấp
- **Tenure**: Vừa phải (≈ 4.5 năm)
- **Đặc điểm khác**:
  - Độ tuổi cao hơn (40–55 tuổi).
  - Tài sản thấp hoặc trung bình.
  - Ít tương tác và phản hồi.
- **Chiến lược gợi ý**:
  - Gửi email “chúng tôi nhớ bạn”, kết hợp phiếu giảm giá hoặc ưu đãi giới hạn thời gian.
  - Khảo sát lý do ngưng sử dụng sản phẩm/dịch vụ.
  - Cung cấp gói khuyến khích quay lại (free shipping, voucher).

#### 🟨 Nhóm 3 – Khách hàng giá trị cao (High-Value Customers)

- **Recency**: Thấp (mua gần đây)
- **Frequency**: Cao nhất (≈ 7.3 đơn hàng)
- **Monetary**: Cao nhất (≈ 4,401 AUD)
- **Tenure**: Cao nhất (≈ 13.5 năm)
- **Đặc điểm khác**:
  - Tuổi trung bình cao (~44 tuổi).
  - Sống ở khu vực có chỉ số tài sản cao (Wealth Segment A).
  - Ưa chuộng sản phẩm chất lượng cao và thương hiệu lớn.
- **Chiến lược gợi ý**:
  - Triển khai chương trình VIP (ưu đãi độc quyền, chăm sóc cá nhân).
  - Mời tham gia khảo sát, review sản phẩm mới.
  - Upsell sản phẩm cao cấp hoặc gói thành viên dài hạn.

>  **Phân tích phân khúc giúp doanh nghiệp xây dựng chiến lược CRM và marketing chính xác hơn**, tập trung đúng đối tượng, tối ưu chi phí và tăng Customer Lifetime Value (CLV).

### 4. Phân tích Khám phá Dữ liệu (Exploratory Data Analysis)

#### 4.1. Hiệu quả kinh doanh theo vùng địa lý

- **New South Wales (NSW)** chiếm hơn **50% tổng doanh thu và lượng khách hàng**, kế đến là:
  - **Victoria (VIC)**: ~25%
  - **Queensland (QLD)**: ~21%
- **Chỉ số ROS (Return on Sales)** và **giá trị trung bình mỗi đơn hàng** giữa các bang dao động quanh mức 2.01, không có sự khác biệt lớn.

##### Xu hướng theo mùa:
- **QLD** đạt doanh thu cao nhất vào **tháng 4–5**
- **VIC** đỉnh vào **tháng 7–8**
- **NSW** đỉnh vào **tháng 8–9**

> *Chiến lược marketing nên được tùy chỉnh theo đặc điểm mùa vụ của từng bang để tối ưu hóa doanh thu.*

#### 4.2. Phân tích hiệu suất theo kênh bán hàng

- **Kênh Online**:
  - Trung bình: **823 đơn/tháng**
  - Thấp hơn mục tiêu 3.06%
- **Kênh Offline**:
  - Trung bình: **810 đơn/tháng**
  - Vượt mục tiêu 1.38%

> *Gợi ý: Tăng cường ưu đãi trực tuyến để thu hút khách và đạt KPI bền vững hơn.*

#### 4.3. Hành vi mua hàng qua phân tích RFM

##### Các chiều phân tích:
- **Recency vs Frequency**
  - *Loyal / High-value*: Gần đây và thường xuyên mua hàng.
  - *Inactive*: Lâu không mua, tần suất thấp.
  - *New*: Mới mua lần đầu.
- **Recency vs Monetary**
  - *High-value*: Gần đây, chi tiêu cao.
  - *Inactive*: Lâu rồi chưa mua, chi tiêu thấp.
  - *New*: Mới mua, chi tiêu thấp.
- **Frequency vs Monetary**
  - *High-value*: Mua nhiều, chi tiêu nhiều.
  - *Loyal*: Mua thường xuyên, chi tiêu trung bình.
  - *New / Inactive*: Thấp cả hai.

> *RFM hỗ trợ phân khúc khách hàng rõ ràng theo hành vi để cá nhân hóa chiến lược chăm sóc.*
#### 4.4. Cohort Analysis – Phân tích gắn bó theo thời gian

- Nhóm khách hàng gia nhập vào tháng **7** và **9** có tỷ lệ giữ chân thấp nhất (giảm từ 1.0 xuống ~0.30–0.39).
- Nhóm khách hàng tháng **1, 2, 5, 8** có tỷ lệ giữ chân tốt hơn và ổn định hơn.

> *Doanh nghiệp nên đẩy mạnh chiến dịch giữ chân vào các tháng có nguy cơ rời bỏ cao, đặc biệt trong mùa đông.*

### 5. Mô hình hóa & Dự đoán khách hàng mới

#### 5.1. Mô hình dữ liệu và Pipeline
- Dữ liệu được chia thành tập huấn luyện (80%) và kiểm tra (20%).
- Sử dụng `ColumnTransformer` để chuẩn hóa và mã hóa đặc trưng (standardization cho biến số, One-Hot Encoding cho biến phân loại).
- Tối ưu hóa hyperparameter thông qua `GridSearchCV` (cho Logistic Regression, MLP) và `RandomizedSearchCV` (cho Random Forest, XGBoost, HistGradientBoosting).


#### 5.2. Các mô hình sử dụng
| Mô hình                     | Kỹ thuật tối ưu             | Cross-validation Score |
|-----------------------------|------------------------------|-------------------------|
| Logistic Regression         | GridSearchCV                 | 0.7563                  |
| Random Forest               | RandomizedSearchCV           | 0.9718                  |
| HistGradientBoosting        | RandomizedSearchCV           | 0.9820                  |
| XGBoost                     | RandomizedSearchCV           | 0.9586                  |
| MLP Classifier              | GridSearchCV                 | 0.9738                  |



#### 5.3. Kết quả huấn luyện các mô hình

| Mô hình                     | Accuracy (Train) | Accuracy (Test) | CV Score |
|-----------------------------|------------------|------------------|----------|
| Logistic Regression         | 75.64%           | 75.40%           | 75.63%   |
| Random Forest               | 99.40%           | 98.03%           | 97.18%   |
| HistGradientBoosting        | 99.92%           | 99.02%           | 98.20%   |
| XGBoost                     | 98.71%           | 96.33%           | 95.86%   |
| MLP Classifier              | 99.22%           | 97.93%           | 97.38%   |

**Nhận xét**:
- HistGradientBoosting và Random Forest đạt hiệu suất cao, không bị overfitting.
- XGBoost có dấu hiệu overfitting nhẹ.
- Logistic Regression cho kết quả thấp nhất, phù hợp với mô hình baseline.



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


#### 5.5. Đánh giá mô hình bằng ROC & Classification Report

- **HistGradientBoosting** và **Random Forest** có AUC gần 1.000 cho tất cả các lớp → mô hình ổn định, tổng quát tốt.
- **XGBoost** đạt AUC cao ở lớp 0 và 1, giảm nhẹ ở lớp 2 và 3.
- **MLP Classifier** hoạt động tốt ở lớp phổ biến, nhưng có sự giảm nhẹ ở lớp ít dữ liệu.
- **Logistic Regression** có AUC thấp, đặc biệt là lớp 2 chỉ đạt 0.7452 → hiệu suất kém hơn hẳn.

#### 5.6. Dự đoán khách hàng mới

- Mô hình XGBoost dự đoán hiệu quả các phân khúc khách hàng mới dựa trên hành vi và đặc điểm nhân khẩu học.
- Nhóm khách hàng tiềm năng (high-value) có giá trị tài sản cao và thời gian gắn bó dài, cho thấy tính khả thi trong việc phát triển chiến lược chăm sóc cá nhân hóa.


**Kết luận**:
- **HistGradientBoosting** là lựa chọn tốt nhất trong bài toán phân khúc khách hàng mới.
- Mô hình có độ chính xác cao, khả năng tổng quát tốt và tránh overfitting.

