# 📊 CUSTOMER SEGMENTATION & BUSINESS STRATEGY PROPOSAL BASED ON RFM, PCA, AND COHORT ANALYSIS

This project analyzes transaction and demographic data of customers from a bicycle retail company in Australia to identify key customer segments, track consumption behavior, and propose data-driven marketing strategies.

## 📁 Project Overview
- **Objective**: Segment customers, analyze behaviors, and develop retention and personalized marketing strategies.
- **Data**: Transactions, demographic information, customer addresses, and new customers (2017).
- **Tools Used**: Python, Power BI, PCA, KMeans, RFM, Cohort Analysis.

## 📌 Project Structure

### 1. 🏢 Context
Understanding customer behavior is a key factor in gaining a competitive edge and sustainable growth. This project applies modern data analysis techniques (RFM, PCA, KMeans, Cohort Analysis) to identify customer segments, monitor retention, and support strategic decision-making.

### 2. Data Cleaning

**Processed datasets:**

#### a. `Transactions`
- Removed duplicates;
- Dropped rows with missing values (`brand`, `product_line`, `standard_cost`);
- Converted `transaction_date` to datetime format;
- Calculated profit margin (`profit_margin`).

#### b. `Customer Demographic`
- Replaced missing values in `job_title`, `industry_category` with “Others”;
- Removed customers with unrealistic `DOB` (e.g., year 1843) and `gender = "U"`;
- Added variables: `customer_age`, `customer_value_score`.

#### c. `Customer Address`
- Merged with demographic table using `customer_id`;
- Used `postcode` and `property_valuation` for geographic segmentation.

#### d. `New Customer List`
- Cleaned similarly to the demographic table;
- Removed rows with incorrect formats;
- Standardized for segmentation prediction.

### 3. Customer Segmentation

After processing and calculating RFM (Recency – Frequency – Monetary), the following steps were performed:
- Normalized RFM and demographic attributes.
- Applied **PCA** to reduce dimensionality and improve clustering efficiency.
- Used **KMeans** (k = 4, determined via Elbow Method and validated with TSNE) for customer segmentation.

This resulted in **4 key customer segments**, each with distinct behavioral and demographic characteristics:

#### 🟦 Cluster 0 – Loyal Customers
- **Recency**: Moderate (≈ 55 days)  
- **Frequency**: High (≈ 6 orders)  
- **Monetary**: Moderate  
- **Tenure**: High (≈ 11.7 years)  
- **Other traits**:
  - Well-distributed across regions;
  - Mostly middle-aged (~35–45);
  - Moderate-to-high asset value.
- **Suggested Strategy**:
  - Loyalty programs, appreciation gifts, periodic discounts.
  - Personalized email marketing based on behavior.

#### 🟩 Cluster 1 – New Potential Customers
- **Recency**: Very recent  
- **Frequency**: Low  
- **Monetary**: Low  
- **Tenure**: Very low (≈ <1 year)  
- **Other traits**:
  - Mostly young adults (~20–30);
  - Early-stage engagement;
  - Evenly spread across online and offline channels.
- **Suggested Strategy**:
  - Onboarding support and usage guidance;
  - First-time buyer offers, repeat purchase incentives;
  - Convert to loyal customers with follow-ups.

#### 🟥 Cluster 2 – Inactive Customers
- **Recency**: High (long time since last purchase)  
- **Frequency**: Very low  
- **Monetary**: Very low  
- **Tenure**: Moderate (≈ 4.5 years)  
- **Other traits**:
  - Older age group (40–55);
  - Low-to-moderate assets;
  - Minimal interaction or feedback.
- **Suggested Strategy**:
  - "We miss you" emails with limited-time offers;
  - Surveys to understand drop-off reasons;
  - Comeback packages (free shipping, vouchers).

#### 🟨 Cluster 3 – High-Value Customers
- **Recency**: Recent  
- **Frequency**: Highest (≈ 7.3 orders)  
- **Monetary**: Highest (≈ 4,401 AUD)  
- **Tenure**: Longest (≈ 13.5 years)  
- **Other traits**:
  - Average age ~44;
  - Reside in high-asset areas (Wealth Segment A);
  - Prefer premium brands and products.
- **Suggested Strategy**:
  - VIP programs, exclusive deals, personalized services;
  - Invite for surveys, product testing;
  - Upsell premium or long-term membership plans.

> **Segmentation enables precise CRM and marketing**, targeting the right customers, optimizing cost, and increasing Customer Lifetime Value (CLV).

### 4. Exploratory Data Analysis (EDA)

#### 4.1. Business Performance by Region
- **New South Wales (NSW)** contributes over **50% of total revenue and customers**, followed by:
  - **Victoria (VIC)**: ~25%
  - **Queensland (QLD)**: ~21%
- **ROS (Return on Sales)** and **average order value** are consistent across states (~2.01).

##### Seasonal Trends:
- **QLD** peaks in **April–May**
- **VIC** in **July–August**
- **NSW** in **August–September**

> *Marketing campaigns should be tailored to each state's seasonal sales pattern.*

#### 4.2. Channel Performance Analysis
- **Online Channel**:
  - Average: **823 orders/month**
  - 3.06% below target
- **Offline Channel**:
  - Average: **810 orders/month**
  - 1.38% above target

> *Recommendation: Boost online incentives to attract and retain customers.*

#### 4.3. RFM-Based Behavioral Analysis

##### Analysis Dimensions:
- **Recency vs Frequency**
  - *Loyal / High-value*: Recent and frequent buyers.
  - *Inactive*: Long inactive, low frequency.
  - *New*: First-time buyers.
- **Recency vs Monetary**
  - *High-value*: Recent, high spending.
  - *Inactive*: Old purchases, low spending.
  - *New*: Recent, low spending.
- **Frequency vs Monetary**
  - *High-value*: Frequent and high spending.
  - *Loyal*: Frequent but moderate spending.
  - *New / Inactive*: Low both.

> *RFM enables precise behavioral segmentation for personalized strategies.*

#### 4.4. Cohort Analysis – Retention over Time
- Customers joining in **July** and **September** showed the lowest retention rates (declining from 1.0 to ~0.30–0.39).
- Customers from **January, February, May, August** had better and more stable retention.

> *Focus retention campaigns in high churn months, especially during winter.*

### 5. Modeling & New Customer Prediction

#### 5.1. Data Modeling & Pipeline
- Data split: 80% training, 20% testing.
- Used `ColumnTransformer` for scaling numerical and encoding categorical features.
- Optimized hyperparameters via `GridSearchCV` (Logistic Regression, MLP) and `RandomizedSearchCV` (Random Forest, XGBoost, HistGradientBoosting).

#### 5.2. Models Used

| Model                    | Optimization Method        | Cross-validation Score |
|--------------------------|----------------------------|-------------------------|
| Logistic Regression      | GridSearchCV               | 0.7563                  |
| Random Forest            | RandomizedSearchCV         | 0.9718                  |
| HistGradientBoosting     | RandomizedSearchCV         | 0.9820                  |
| XGBoost                  | RandomizedSearchCV         | 0.9586                  |
| MLP Classifier           | GridSearchCV               | 0.9738                  |

#### 5.3. Model Performance

| Model                    | Accuracy (Train) | Accuracy (Test) | CV Score |
|--------------------------|------------------|------------------|----------|
| Logistic Regression      | 75.64%           | 75.40%           | 75.63%   |
| Random Forest            | 99.40%           | 98.03%           | 97.18%   |
| HistGradientBoosting     | 99.92%           | 99.02%           | 98.20%   |
| XGBoost                  | 98.71%           | 96.33%           | 95.86%   |
| MLP Classifier           | 99.22%           | 97.93%           | 97.38%   |

**Remarks**:
- HistGradientBoosting and Random Forest show high accuracy without overfitting.
- XGBoost shows slight overfitting.
- Logistic Regression performs the weakest, serving as a baseline model.

#### 5.4. Hyperparameter Optimization

(You can copy-paste the detailed hyperparameter table from the previous section here as needed.)

#### 5.5. Model Evaluation via ROC & Classification Report
- HistGradientBoosting and Random Forest: AUC near 1.000 across all classes.
- XGBoost: Slightly lower AUC in less frequent classes.
- Logistic Regression: Weakest performance, especially in class 2.

#### 5.6. New Customer Prediction
- XGBoost accurately predicts new customer segments based on behavior and demographics.
- High-value predicted customers exhibit high asset values and long-term potential.

**Conclusion**:
- **HistGradientBoosting** is the most effective model for new customer segmentation.
- It shows high accuracy, strong generalization, and minimal overfitting.
