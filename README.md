
# House Price Prediction: Advanced Feature Engineering & Selection

## 🎯 Objective
The goal of this project is to enhance the predictive power of a housing dataset through strategic feature engineering and selection techniques. By transforming raw variables into meaningful domain-driven metrics, analyzing historical construction eras, and mathematically isolating the most impactful predictors, this pipeline optimizes data readiness for complex regression models.

## 📊 Dataset Information
This project utilizes the **Ames Housing Dataset** (`House_Prices.csv`). It contains 1,460 observations and 81 features detailing nearly every aspect of residential homes in Ames, Iowa. The target variable is `SalePrice`.

## 🛠️ Tasks & Outcomes

### 1. Domain-Driven Feature Creation
*   **Action:** Synthesized fundamental housing metrics from fragmented raw data.
*   **Outcome:** Created `TotalSF` (Total Square Footage), `HouseAge` (Years since construction), and `TotalBath` (Aggregated full and half baths).

### 2. Polynomial Feature Expansion
*   **Action:** Explored non-linear relationships and amplified the signal of highly correlated features.
*   **Outcome:** Generated squared terms (`TotalSF^2`, `OverallQual^2`) and interaction terms (`TotalSF * OverallQual`).

### 3. Binning & Discretization
*   **Action:** Grouped continuous historical timelines to make the model robust against outliers and historical anomalies.
*   **Outcome:** Segmented `YearBuilt` into distinct categorical eras: `Pre-1950`, `1950-1980`, `1980-2000`, and `2000+`.

### 4. Target Encoding
*   **Action:** Converted complex categorical variables into a continuous numeric scale correlated directly with the target.
*   **Outcome:** Transformed the `Neighborhood` category into `Neighborhood_Target_Enc` based on historical average `SalePrice`.

### 5. Advanced Feature Selection
*   **Mutual Information (MI):** Evaluated non-linear statistical dependencies to rank standalone feature importance.
*   **Recursive Feature Elimination (RFE):** Iteratively pruned the weakest features using a Linear Regression estimator.
*   **L1 Regularization (Lasso):** Applied a penalty factor mathematically shrinking less critical coefficients exactly to zero.
*   **Outcome:** Successfully isolated the top 5–8 most vital predictive features required for model deployment.

## 🧠 Key Skills Demonstrated
*   **Data Manipulation:** `pandas`, `numpy`
*   **Feature Engineering:** `sklearn.preprocessing` (PolynomialFeatures, Binning)
*   **Category Transformation:** Target Encoding for nominal variables.
*   **Feature Selection:** Mutual Information Regression, RFE (Recursive Feature Elimination), L1/Lasso Regularization.
*   **Data Visualization:** `matplotlib.pyplot`, `seaborn`

## 💡 Key Learnings
*   **Simplicity Wins:** Aggregating raw, fragmented features into logical domain metrics (like `TotalSF`) often yields higher predictive power than allowing an algorithm to interpret raw square footage room-by-room.
*   **Curse of Dimensionality:** Expanding data via polynomial features is highly effective but demands immediate application of feature selection (like Lasso or RFE) to prevent model overfitting.
*   **Encoding Strategy Matters:** Traditional One-Hot Encoding on high-cardinality features like `Neighborhood` explodes the dataset width. Target Encoding successfully captured historical neighborhood value continuously without adding a single extra column.
