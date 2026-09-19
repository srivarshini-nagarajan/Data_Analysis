# 1. IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)
# 3. TITLE
st.title("🛒 E-Commerce Sales & Customer Analytics Dashboard")

st.write(
    "This dashboard provides an easy-to-understand view of "
    "sales, customers, orders, profits and repeat customers."
)
# 4. LOAD DATA
FILE_PATH = (
    "ecommerce_sales_customer_analytics_.csv"
)
@st.cache_data
def load_data():
    data = pd.read_csv(FILE_PATH)
    # Remove completely empty rows
    data = data.dropna(how="all")
    return data
data = load_data()
# 5. BASIC DATA CLEANING
# Convert important numeric columns into numbers
numeric_columns = [
    "customer_age",
    "delivery_days",
    "estimated_delivery_days",
    "customer_rating",
    "loyalty_points_earned",
    "loyalty_points_redeemed",
    "quantity",
    "gross_sales",
    "discount_amount",
    "tax_amount",
    "shipping_cost",
    "net_sales",
    "product_cost",
    "profit",
    "profit_margin_percentage",
    "customer_lifetime_value",
    "customer_order_count",
    "is_repeat_customer"
]
for column in numeric_columns:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )
# 6. FIND DEVICE TYPE COLUMN
# The current CSV may not contain a device column.
# This code checks several possible column names.
device_column = None
possible_device_columns = [
    "device_type",
    "device",
    "customer_device",
    "deviceType"
]
for column in possible_device_columns:
    if column in data.columns:
        device_column = column
        break
# 7. SIDEBAR FILTERS
st.sidebar.header("🔎 Dashboard Filters")
# Country Filter
if "customer_country" in data.columns:
    country_options = sorted(
        data["customer_country"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    selected_country = st.sidebar.multiselect(
        "🌍 Country",
        options=country_options,
        default=country_options
    )
else:
    selected_country = []
# City Filter
if "customer_city" in data.columns:
    city_options = sorted(
        data["customer_city"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    selected_city = st.sidebar.multiselect(
        "🏙️ City",
        options=city_options,
        default=city_options
    )
else:
    selected_city = []
# Device Type Filter
if device_column is not None:
    device_options = sorted(
        data[device_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    selected_device = st.sidebar.multiselect(
        "📱 Device Type",
        options=device_options,
        default=device_options
    )
else:
    st.sidebar.info(
        "Device Type column was not found in the CSV."
    )
    selected_device = []
# Region Filter
if "region" in data.columns:
    region_options = sorted(
        data["region"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    selected_region = st.sidebar.multiselect(
        "🌎 Region",
        options=region_options,
        default=region_options
    )
else:
    selected_region = []
# Customer Segment Filter
if "customer_segment" in data.columns:
    segment_options = sorted(
        data["customer_segment"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    selected_segment = st.sidebar.multiselect(
        "👥 Customer Segment",
        options=segment_options,
        default=segment_options
    )
else:
    selected_segment = []
# Sales Channel Filter
if "sales_channel" in data.columns:
    channel_options = sorted(
        data["sales_channel"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    selected_channel = st.sidebar.multiselect(
        "🛍️ Sales Channel",
        options=channel_options,
        default=channel_options
    )
else:
    selected_channel = []
# Net Sales Filter
if "net_sales" in data.columns:
    min_sales = float(
        data["net_sales"].min()
    )
    max_sales = float(
        data["net_sales"].max()
    )
    selected_sales = st.sidebar.slider(
        "💰 Net Sales Range",
        min_value=min_sales,
        max_value=max_sales,
        value=(min_sales, max_sales)
    )
else:
    selected_sales = (0, 0)
# Discount Filter
if "discount_amount" in data.columns:
    min_discount = float(
        data["discount_amount"].min()
    )
    max_discount = float(
        data["discount_amount"].max()
    )
    selected_discount = st.sidebar.slider(
        "🏷️ Discount Amount",
        min_value=min_discount,
        max_value=max_discount,
        value=(min_discount, max_discount)
    )
else:
    selected_discount = (0, 0)
# Customer Rating Filter
if "customer_rating" in data.columns:
    min_rating = float(
        data["customer_rating"].min()
    )
    max_rating = float(
        data["customer_rating"].max()
    )
    selected_rating = st.sidebar.slider(
        "⭐ Customer Rating",
        min_value=min_rating,
        max_value=max_rating,
        value=(min_rating, max_rating)
    )
else:
    selected_rating = (0, 0)
# 8. APPLY FILTERS
filtered_data = data.copy()
# Country
if "customer_country" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["customer_country"]
        .astype(str)
        .isin(selected_country)
    ]
# City
if "customer_city" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["customer_city"]
        .astype(str)
        .isin(selected_city)
    ]
# Device
if device_column is not None:
    filtered_data = filtered_data[
        filtered_data[device_column]
        .astype(str)
        .isin(selected_device)
    ]
# Region
if "region" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["region"]
        .astype(str)
        .isin(selected_region)
    ]
# Customer Segment
if "customer_segment" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["customer_segment"]
        .astype(str)
        .isin(selected_segment)
    ]
# Sales Channel
if "sales_channel" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["sales_channel"]
        .astype(str)
        .isin(selected_channel)
    ]
# Net Sales
if "net_sales" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["net_sales"].between(
            selected_sales[0],
            selected_sales[1]
        )
    ]
# Discount
if "discount_amount" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["discount_amount"].between(
            selected_discount[0],
            selected_discount[1]
        )
    ]
# Rating
if "customer_rating" in filtered_data.columns:
    filtered_data = filtered_data[
        filtered_data["customer_rating"].between(
            selected_rating[0],
            selected_rating[1]
        )
    ]
# 9. CHECK FILTERED DATA
if filtered_data.empty:
    st.warning(
        "⚠️ No data is available for the selected filters."
    )
    st.stop()
st.success(
    f"Showing {len(filtered_data):,} records"
)
# 10. BUSINESS OVERVIEW
st.header("📊 Business Overview")
# Calculate metrics
total_sales = filtered_data["net_sales"].sum()
total_profit = filtered_data["profit"].sum()
total_orders = (
    filtered_data["order_id"].nunique()
    if "order_id" in filtered_data.columns
    else len(filtered_data)
)
average_rating = (
    filtered_data["customer_rating"].mean()
    if "customer_rating" in filtered_data.columns
    else 0
)
# Create four columns
col1, col2, col3, col4 = st.columns(4)
# Total Sales in Millions
with col1:
    total_sales_million = total_sales / 1_000_000
    st.metric(
        "💰 Total Sales",
        f"{total_sales_million:.2f}M"
    )
# Total Profit
with col2:
    st.metric(
        "📈 Total Profit",
        f"{total_profit:,.2f}"
    )
# Total Orders
with col3:
    st.metric(
        "📦 Total Orders",
        f"{total_orders:,}"
    )
# Average Rating
with col4:
    st.metric(
        "⭐ Average Rating",
        f"{average_rating:.2f}"
    )
# 11. CHART 1 - SALES BY COUNTRY
if "customer_country" in filtered_data.columns:
    st.subheader("🌍 Sales by Country")
    country_sales = (
        filtered_data
        .groupby("customer_country")["net_sales"]
        .sum()
        .reset_index()
        .sort_values(
            "net_sales",
            ascending=False
        )
    )
    fig_country = px.bar(
        country_sales,
        x="customer_country",
        y="net_sales",
        title="Total Sales by Country",
        labels={
            "customer_country": "Country",
            "net_sales": "Net Sales"
        },
        color="customer_country",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    fig_country.update_layout(
        xaxis_title="Country",
        yaxis_title="Sales",
        showlegend=False
    )
    st.plotly_chart(
        fig_country,
        use_container_width=True
    )
# 12. CHART 2 - SALES BY CITY
if "customer_city" in filtered_data.columns:
    st.subheader("🏙️ Sales by City")
    city_sales = (
        filtered_data
        .groupby("customer_city")["net_sales"]
        .sum()
        .reset_index()
        .sort_values(
            "net_sales",
            ascending=False
        )
        .head(15)
    )
    fig_city = px.bar(
        city_sales,
        x="customer_city",
        y="net_sales",
        title="Top 15 Cities by Sales",
        labels={
            "customer_city": "City",
            "net_sales": "Net Sales"
        },
        color="customer_city",
        color_discrete_sequence=px.colors.sequential.Greens
    )
    fig_city.update_layout(
        xaxis_title="City",
        yaxis_title="Sales",
        showlegend=False
    )
    st.plotly_chart(
        fig_city,
        use_container_width=True
    )
# 13. CHART 3 - SALES BY REGION
if "region" in filtered_data.columns:
    st.subheader("🌎 Sales by Region")
    region_sales = (
        filtered_data
        .groupby("region")["net_sales"]
        .sum()
        .reset_index()
        .sort_values(
            "net_sales",
            ascending=False
        )
    )
    fig_region = px.bar(
        region_sales,
        x="region",
        y="net_sales",
        title="Total Sales by Region",
        labels={
            "region": "Region",
            "net_sales": "Net Sales"
        },
        color="region",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_region.update_layout(
        showlegend=False
    )
    st.plotly_chart(
        fig_region,
        use_container_width=True
    )
# 14. CHART 4 - CUSTOMER SEGMENT
if "customer_segment" in filtered_data.columns:
    st.subheader("👥 Sales by Customer Segment")
    segment_sales = (
        filtered_data
        .groupby("customer_segment")["net_sales"]
        .sum()
        .reset_index()
    )
    fig_segment = px.pie(
        segment_sales,
        names="customer_segment",
        values="net_sales",
        title="Sales Distribution by Customer Segment",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(
        fig_segment,
        use_container_width=True
    )
# 15. CHART 5 - SALES CHANNEL
if "sales_channel" in filtered_data.columns:
    st.subheader("🛍️ Sales by Sales Channel")
    channel_sales = (
        filtered_data
        .groupby("sales_channel")["net_sales"]
        .sum()
        .reset_index()
    )
    fig_channel = px.bar(
        channel_sales,
        x="sales_channel",
        y="net_sales",
        title="Sales by Sales Channel",
        labels={
            "sales_channel": "Sales Channel",
            "net_sales": "Net Sales"
        },
        color="sales_channel",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_channel.update_layout(
        showlegend=False
    )
    st.plotly_chart(
        fig_channel,
        use_container_width=True
    )
# 16. CHART 6 - PAYMENT METHOD
if "payment_method" in filtered_data.columns:
    st.subheader("💳 Sales by Payment Method")
    payment_sales = (
        filtered_data
        .groupby("payment_method")["net_sales"]
        .sum()
        .reset_index()
    )
    fig_payment = px.bar(
        payment_sales,
        x="payment_method",
        y="net_sales",
        title="Sales by Payment Method",
        labels={
            "payment_method": "Payment Method",
            "net_sales": "Net Sales"
        },
        color="payment_method",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_payment.update_layout(
        showlegend=False
    )
    st.plotly_chart(
        fig_payment,
        use_container_width=True
    )
# 17. CHART 7 - DISCOUNT VS PROFIT
if (
    "discount_amount" in filtered_data.columns
    and "profit" in filtered_data.columns
):
    st.subheader("🏷️ Discount Amount vs Profit")
    fig_discount = px.scatter(
        filtered_data,
        x="discount_amount",
        y="profit",
        title="Discount Amount vs Profit",
        labels={
            "discount_amount": "Discount Amount",
            "profit": "Profit"
        },
        color="profit",
        color_continuous_scale="Plasma"
    )
    st.plotly_chart(
        fig_discount,
        use_container_width=True
    )
# 18. CHART 8 - REPEAT CUSTOMER DISTRIBUTION
if "is_repeat_customer" in filtered_data.columns:
    st.subheader("🔄 Repeat Customer Distribution")
    repeat_data = (
        filtered_data["is_repeat_customer"]
        .value_counts()
        .reset_index()
    )
    repeat_data.columns = [
        "is_repeat_customer",
        "count"
    ]
    repeat_data["Customer Type"] = (
        repeat_data["is_repeat_customer"]
        .map({
            0: "New Customer",
            1: "Repeat Customer"
        })
    )
    fig_repeat = px.pie(
        repeat_data,
        names="Customer Type",
        values="count",
        title="New Customers vs Repeat Customers",
        color="Customer Type",
        color_discrete_map={
            "New Customer": "#FF9999",
            "Repeat Customer": "#66B3FF"
        }
    )
    st.plotly_chart(
        fig_repeat,
        use_container_width=True
    )
# 19. STATISTICAL SUMMARY
st.header("📋 Statistical Summary")
st.dataframe(
    filtered_data.describe(),
    use_container_width=True
)
# 20. MACHINE LEARNING
st.header("🤖 Customer Repeat Purchase Prediction")
st.write(
    "The Logistic Regression model is trained automatically "
    "to predict whether a customer is a repeat customer."
)
# Check required columns
required_ml_columns = [
    "customer_age",
    "quantity",
    "net_sales",
    "profit",
    "customer_lifetime_value",
    "customer_order_count",
    "is_repeat_customer"
]
missing_ml_columns = [
    column
    for column in required_ml_columns
    if column not in filtered_data.columns
]
if len(missing_ml_columns) > 0:
    st.warning(
        "The following columns are required for machine "
        "learning but are missing: "
        + ", ".join(missing_ml_columns)
    )
else:
    # Prepare ML data
    ml_data = filtered_data[
        required_ml_columns
    ].dropna()
    # Check enough data
    if len(ml_data) < 10:
        st.warning(
            "Not enough data is available to train the model."
        )
    elif ml_data["is_repeat_customer"].nunique() < 2:
        st.warning(
            "The filtered data contains only one customer "
            "class. Please select broader filters."
        )
    else:
        X = ml_data[
            [
                "customer_age",
                "quantity",
                "net_sales",
                "profit",
                "customer_lifetime_value",
                "customer_order_count"
            ]
        ]
        y = ml_data["is_repeat_customer"]
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train
        )
        X_test_scaled = scaler.transform(
            X_test
        )
        # Train Logistic Regression
        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        model.fit(
            X_train_scaled,
            y_train
        )
        # Predictions
        y_pred = model.predict(
            X_test_scaled
        )
        # Model Accuracy
        accuracy = accuracy_score(
            y_test,
            y_pred
        )
        st.subheader("📈 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Model Accuracy",
                f"{accuracy * 100:.2f}%"
            )
        with col2:
            st.metric(
                "Training Records",
                f"{len(X_train):,}"
            )
        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(
            y_test,
            y_pred
        )
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            title="Confusion Matrix",
            labels={
                "x": "Predicted",
                "y": "Actual"
            },
            color_continuous_scale="Blues"
        )
        fig_cm.update_xaxes(
            ticktext=[
                "New Customer",
                "Repeat Customer"
            ],
            tickvals=[0, 1]
        )
        fig_cm.update_yaxes(
            ticktext=[
                "New Customer",
                "Repeat Customer"
            ],
            tickvals=[0, 1]
        )
        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )
        # Classification Report
        st.subheader("📊 Classification Report")
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(
            report
        ).transpose()
        st.dataframe(
            report_df,
            use_container_width=True
        )
        # 21. CUSTOMER PREDICTION FORM
        st.subheader(
            "🔮 Predict Customer Type"
        )
        st.write(
            "Enter customer information below to predict "
            "whether the customer is likely to be a repeat customer."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            prediction_age = st.number_input(
                "Customer Age",
                min_value=1,
                max_value=100,
                value=30
            )
        with col2:
            prediction_quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=2
            )
        with col3:
            prediction_sales = st.number_input(
                "Net Sales",
                min_value=0.0,
                value=1000.0
            )
        col1, col2, col3 = st.columns(3)
        with col1:
            prediction_profit = st.number_input(
                "Profit",
                value=200.0
            )
        with col2:
            prediction_lifetime = st.number_input(
                "Customer Lifetime Value",
                min_value=0.0,
                value=5000.0
            )


        with col3:

            prediction_orders = st.number_input(
                "Customer Order Count",
                min_value=1,
                value=2
            )
        # Make prediction automatically
        prediction_data = pd.DataFrame(
            {
                "customer_age": [
                    prediction_age
                ],
                "quantity": [
                    prediction_quantity
                ],
                "net_sales": [
                    prediction_sales
                ],
                "profit": [
                    prediction_profit
                ],
                "customer_lifetime_value": [
                    prediction_lifetime
                ],
                "customer_order_count": [
                    prediction_orders
                ]
            }
        )
        prediction_scaled = scaler.transform(
            prediction_data
        )
        prediction = model.predict(
            prediction_scaled
        )[0]
        probability = model.predict_proba(
            prediction_scaled
        )[0]
        # Display prediction
        if prediction == 1:
            st.success(
                "🔄 Prediction: Repeat Customer"
            )
        else:
            st.info(
                "🆕 Prediction: New Customer"
            )
        st.write(
            f"Probability of Repeat Customer: "
            f"{probability[1] * 100:.2f}%"
        )
# 22. FOOTER
st.markdown("---")
st.caption(
    "E-Commerce Sales & Customer Analytics Dashboard "
    "| Built with Streamlit, Pandas, Plotly and Scikit-learn"
)
