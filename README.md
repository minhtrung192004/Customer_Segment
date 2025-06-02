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

### 5. 🤖 Dự Đoán Phân Khúc Cho Khách Hàng Mới

- **Xử lý đặc trưng**: One-hot encoding, scaling dữ liệu.
- **Huấn luyện mô hình**:
  - Logistic Regression, Random Forest, XGBoost, MLP
  - Đánh giá: Accuracy, F1-score, AUC
- **Kết quả**:
  - Dự đoán nhóm khách hàng tiềm năng từ dữ liệu mới.
  - Giải thích độ quan trọng của biến bằng SHAP (XGBoost)
