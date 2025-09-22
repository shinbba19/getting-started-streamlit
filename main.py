import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import altair as alt

# --------------------------------------------------
# Sidebar Menu
# --------------------------------------------------
menu = ["Data Dashboard", "Price Prediction"]
choice = st.sidebar.selectbox("Menu", menu)

# --------------------------------------------------
# 1) Data Dashboard
# --------------------------------------------------
if choice == "Data Dashboard":
    st.header("📊 Data Dashboard (สำรวจข้อมูลรวม)")

    # Load dataset
    df = pd.read_csv("Price per sqm_cleaned_data_selection2.csv")

    # Sidebar Filters
    st.sidebar.subheader("🔍 ตัวกรองข้อมูล")
    district_options = sorted(df["district_type"].unique())
    selected_district = st.sidebar.multiselect(
        "เลือกโซนทำเล (district_type)", district_options, default=district_options
    )
    age_min, age_max = int(df["bld_age"].min()), int(df["bld_age"].max())
    age_range = st.sidebar.slider("ช่วงอายุอาคาร (ปี)", age_min, age_max, (age_min, age_max))
    price_min, price_max = int(df["price_sqm"].min()), int(df["price_sqm"].max())
    price_range = st.sidebar.slider("ช่วงราคาต่อตร.ม.", price_min, price_max, (price_min, price_max))

    # Apply Filters
    filtered_df = df[
        (df["district_type"].isin(selected_district)) &
        (df["bld_age"].between(age_range[0], age_range[1])) &
        (df["price_sqm"].between(price_range[0], price_range[1]))
    ]

    # KPI Summary
    st.subheader("📌 สรุปข้อมูล (KPI Overview)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ราคาเฉลี่ย (฿/ตร.ม.)", f"{filtered_df['price_sqm'].mean():,.0f}")
    with col2:
        st.metric("ราคาสูงสุด (฿/ตร.ม.)", f"{filtered_df['price_sqm'].max():,.0f}")
    with col3:
        st.metric("ราคาต่ำสุด (฿/ตร.ม.)", f"{filtered_df['price_sqm'].min():,.0f}")
    with col4:
        st.metric("อายุอาคารเฉลี่ย (ปี)", f"{filtered_df['bld_age'].mean():.1f}")

    st.write("---")

    # Filtered Data
    st.subheader("📋 ข้อมูลที่ถูกกรอง")
    st.write(f"จำนวนแถวที่เลือก: {len(filtered_df)}")
    st.dataframe(filtered_df.head(20))

    # Chart 1: Distribution of Price
    st.subheader("📈 การกระจายราคาต่อตร.ม.")
    chart1 = alt.Chart(filtered_df).mark_bar().encode(
        alt.X("price_sqm", bin=alt.Bin(maxbins=50), title="ราคาต่อตร.ม."),
        y='count()'
    )
    st.altair_chart(chart1, use_container_width=True)

    # Chart 2: Age vs Price
    st.subheader("🏗️ ความสัมพันธ์ระหว่างอายุอาคารกับราคา")
    chart2 = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.5).encode(
        x=alt.X("bld_age", title="อายุอาคาร (ปี)"),
        y=alt.Y("price_sqm", title="ราคาต่อตร.ม."),
        color="district_type:N",
        tooltip=["bld_age", "price_sqm", "district_type"]
    )
    st.altair_chart(chart2, use_container_width=True)

    # Chart 3: Distance to BTS vs Price
    st.subheader("🚉 ระยะทางจาก BTS กับราคาต่อตร.ม.")
    chart3 = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("Aver_trans", title="ระยะทางจาก BTS (กม.)"),
        y=alt.Y("price_sqm", title="ราคาต่อตร.ม."),
        color="district_type:N",
        tooltip=["Aver_trans", "price_sqm", "district_type", "bld_age"]
    )
    st.altair_chart(chart3, use_container_width=True)

    # Chart 4: District vs Average Price
    st.subheader("🏙️ ราคาเฉลี่ยตาม District")
    district_chart = filtered_df.groupby("district_type")["price_sqm"].mean().reset_index()
    chart4 = alt.Chart(district_chart).mark_bar().encode(
        x="district_type:N",
        y="price_sqm:Q"
    )
    st.altair_chart(chart4, use_container_width=True)

    # Chart 5: Facilities Impact (Pool)
    st.subheader("🏊 สิ่งอำนวยความสะดวกกับราคาเฉลี่ย (Pool)")
    facility_chart = filtered_df.groupby("Pool")["price_sqm"].mean().reset_index()
    chart5 = alt.Chart(facility_chart).mark_bar().encode(
        x="Pool:N",
        y="price_sqm:Q"
    )
    st.altair_chart(chart5, use_container_width=True)

# --------------------------------------------------
# 2) Price Prediction
# --------------------------------------------------
elif choice == "Price Prediction":
    st.header("📈 ทำนายราคาทรัพย์ด้วย Random Forest (จาก Dataset จริง)")

    # Load dataset
    df = pd.read_csv("Price per sqm_cleaned_data_selection2.csv")

    # Features
    features = [
        "bld_age", "nbr_floors", "Aver_trans", "district_type",
        "Pool", "Gym", "Parking", "Elevator", "Security",
        "Aver_food", "Aver_school", "Aver_shop", "hospital"
    ]
    df = df.dropna(subset=features + ["price_sqm"])
    X = df[features]
    y = df["price_sqm"]

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader("📊 ประสิทธิภาพของโมเดล")
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**MAE:** {mae:,.0f} บาท/ตร.ม.")
    st.write(f"**RMSE:** {rmse:,.0f} บาท/ตร.ม.")

    # ---------------- User Input ----------------
    st.subheader("กรอกข้อมูลเพื่อทำนาย")
    size = st.number_input("ขนาดห้อง (ตร.ม.)", 20, 200, 35)
    age = st.number_input("อายุอาคาร (ปี)", 0, 40, 10)
    floor = st.number_input("ชั้นที่อยู่", 1, 60, 5)
    distance = st.number_input("ระยะทางจาก BTS (กม.)", 0.0, 15.0, 1.0, step=0.1)
    district = st.selectbox("โซนทำเล (district_type)", sorted(df["district_type"].unique()))

    # Preset options (ค่า default)
    pool, gym, parking, elevator, security = 1, 1, 1, 1, 1
    avg_food = float(df["Aver_food"].mean())
    avg_school = float(df["Aver_school"].mean())
    avg_shop = float(df["Aver_shop"].mean())
    hospital = float(df["hospital"].mean())

    if st.button("ทำนายราคา"):
        X_new = np.array([[
            age, floor, distance, district,
            pool, gym, parking, elevator, security,
            avg_food, avg_school, avg_shop, hospital
        ]])
        pred_price_sqm = model.predict(X_new)[0]
        total_price = pred_price_sqm * size

        st.success(f"🏷️ ราคาต่อตร.ม. ≈ {pred_price_sqm:,.0f} บาท")
        st.success(f"💰 ราคาห้องรวม ≈ {total_price:,.0f} บาท")

        # Feature Importance
        st.subheader("🔎 ความสำคัญของตัวแปร (Feature Importance)")
        fi = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(fi.set_index("Feature"))
