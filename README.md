#  E-Commerce Sales & Customer Analytics Dashboard

An interactive **E-Commerce Sales & Customer Analytics Dashboard** built using **Python, Streamlit, Pandas, Plotly, and Scikit-learn**.
The dashboard helps analyze sales performance, customer behavior, profitability, payment methods, sales channels, and repeat customers through interactive filters and visualizations.

##  Project Overview
This project provides an interactive dashboard for analyzing an e-commerce dataset.
Users can filter the data based on:
-  Country
-  City
-  Device Type (if available in the dataset)
-  Region
-  Customer Segment
-  Sales Channel
-  Net Sales
-  Discount Amount
-  Customer Rating
The dashboard automatically updates the displayed metrics and charts based on the selected filters.

---
##  Streamlit - https://ecommerce-datanalysis.streamlit.app/
##  Features
###  Business Overview
The dashboard displays key business metrics:
-  Total Sales
-  Total Profit
-  Total Orders
-  Average Customer Rating
Total Sales is displayed in **millions** for easier interpretation.
---

###  Interactive Visualizations
The dashboard includes several charts:
1.  Sales by Country
2.  Sales by City
3.  Sales by Region
4.  Sales by Customer Segment
5.  Sales by Sales Channel
6.  Sales by Payment Method
7.  Discount Amount vs Profit
8.  New Customers vs Repeat Customers
Each visualization uses different colors to make the dashboard easy to understand.

---
##  Machine Learning
The project also includes a **Logistic Regression** machine learning model.
The model predicts whether a customer is likely to be a:
-  New Customer
-  Repeat Customer

### Model Features
The prediction model uses:
- Customer Age
- Quantity
- Net Sales
- Profit
- Customer Lifetime Value
- Customer Order Count
  
The model automatically:
- Splits the dataset into training and testing data
- Scales the numerical features
- Trains a Logistic Regression model
- Calculates model accuracy
- Generates a confusion matrix
- Generates a classification report
- Provides customer-level predictions

There is **no separate Train Model button**. The model is trained automatically when the application runs.
