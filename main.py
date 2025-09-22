import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import altair as alt

# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------
if "deals" not in st.session_state:
    st.session_state.deals = []

# --------------------------------------------------
# Sidebar Menu
# --------------------------------------------------
menu = [
    "Home",
    "Owner Dashboard",
    "Investor Dashboard",
    "Deal Summary",
    "Data Dashboard",
    "Price Prediction"
]
choice = st.sidebar.selectbox("Menu", menu)

# --------------------------------------------------
# 1) Home
# --------------------------------------------------
if choice == "Home":
    st.title("🏠 P2P ขายฝาก Platform (Prototype)")
    st.write("""
    Prototype นี้จำลองระบบ P2P ขายฝากแบบง่าย **(ไม่มี blockchain, ไม่ต้อง login)**  

    🔹 Features:
    - เจ้าของทรัพย์: สามารถโพสต์ข้อมูลขายฝาก  
    - นักลงทุน: เลือกลงทุนในดีลที่สนใจ  
    - Dashboard: สำรวจข้อมูลราคาตลาดจริง  
    - Model: ทำนายราคาทรัพย์ด้วย Random Forest  
    """)

# --------------------------------------------------
# 2) Owner Dashboard
# --------------------------------------------------
elif choice == "Owner Dashboard":
    st.header("📌 เจ้าของโพสต์ดีลขายฝาก")
    with st.form("owner_form"):
        property_name = st.text_input("ชื่อทรัพย์สิน (เช่น คอนโดสุขุมวิท 40)")
        location = st.text_input("ทำเล/ที่ตั้ง")
        price = st.number_input("วงเงินขายฝาก (บาท)", min_value=100000, step=50000)
        duration = st.selectbox("ระยะเวลาไถ่ถอน", ["6 เดือน", "1 ปี", "2 ปี"])
        submitted = st.form_submit_button("โพสต์ดีล")

        if submitted:
            deal = {
                "property": property_name,
                "location": location,
                "price": price,
                "duration": duration,
                "status": "รอจับคู่"
            }
            st.session_state.deals.append(deal)
            st.success("✅ โพสต์ดีลสำเร็จ!")

# --------------------------------------------------
# 3) Investor Dashboard
# --------------------------------------------------
elif choice == "Investor Dashboard":
    st.header("💰 นักลงทุนเลือกดีล")
    if len(st.session_state.deals) == 0:
        st.info("ยังไม่มีดีลที่โพสต์")
    else:
        for i, deal in enumerate(st.session_state.deals):
            with st.expander(f"ดีล {i+1}: {deal['property']}"):
                st.write(f"📍 {deal['location']}")
                st.write(f"💵 {deal['price']:,} บาท")
                st.write(f"⏳ {deal['duration']}")
                if st.button(f"ลงทุนในดีล {i+1}", key=f"invest_{i}"):
                    st.session_state.deals[i]["status"] = "มีนักลงทุนแล้ว ✅"
                    st.success(f"คุณเลือกลงทุนใน {deal['property']}")

# --------------------------------------------------
# 4) Deal Summary
# --------------------------------------------------
elif choice == "Deal Summary":
    st.header("📊 สรุปดีลทั้งหมด")
    if len(st.session_state.deals) > 0:
        df = pd.DataFrame(st.session_state.deals)
        st.dataframe(df)
    else:
        st.warning("ยังไม่มีดีลในระบบ")

# --------------------------------------------------
# 5) Data Dashboard
# --------------------------------------------------
elif choice == "Data Dashboard":
    st.header("📊 Data Dashboard (สำรวจข้อมูลรวม)")

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

    # Chart 3: Facilities Impact (Pool)
    st.subheader("🏊 สิ่งอำนวยความสะดวกกับราคาเฉลี่ย")
    facility_chart = filtered_df.groupby("Pool")["price_sqm"].mean().reset_index()
    chart3 = alt.Chart(facility_chart).mark_bar().encode(
        x="Pool:N",
        y="price_sqm:Q"
    )
    st.altair_chart(chart3, use_container_width=True)

# --------------------------------------------------
# 6) Price Prediction
# --------------------------------------------------
elif choice == "Price Prediction":
    st.header("📈 ทำนายราคาทรัพย์ด้วย Random Forest (จาก Dataset จริง)")

    df = pd.read_csv("Price per sqm_cleaned_data_selection2.csv")

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

    # User Input
    st.subheader("กรอกข้อมูลเพื่อทำนาย")
    size = st.number_input("ขนาดห้อง (ตร.ม.)", 20, 200, 35)
    age = st.number_input("อายุอาคาร (ปี)", 0, 40, 10)
    floor = st.number_input("ชั้นที่อยู่", 1, 60, 5)
    distance = st.number_input("ระยะทางจาก BTS (กม.)", 0.0, 15.0, 1.0, step=0.1)

    # Preset Options
    district = st.selectbox("โซนทำเล (district_type)", sorted(df["district_type"].unique()))
    pool = st.checkbox("มีสระว่ายน้ำ", value=True)
    gym = st.checkbox("มียิม", value=True)
    parking = st.checkbox("มีที่จอดรถ", value=True)
    elevator = st.checkbox("มีลิฟต์", value=True)
    security = st.checkbox("มีระบบรักษาความปลอดภัย", value=True)

    avg_food = st.slider("คะแนนร้านอาหารรอบๆ (0-1)", 0.0, 1.0, float(df["Aver_food"].mean()))
    avg_school = st.slider("คะแนนโรงเรียนรอบๆ (0-1)", 0.0, 1.0, float(df["Aver_school"].mean()))
    avg_shop = st.slider("คะแนนร้านค้า/ห้างรอบๆ (0-1)", 0.0, 1.0, float(df["Aver_shop"].mean()))
    hospital = st.slider("ระยะห่างโรงพยาบาล (กม.)", 0.0, 5.0, float(df["hospital"].mean()))

    if st.button("ทำนายราคา"):
        X_new = np.array([[
            age, floor, distance, district,
            int(pool), int(gym), int(parking), int(elevator), int(security),
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
