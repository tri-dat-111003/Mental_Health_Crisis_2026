# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:48:58 2026

@author: trida
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Mental Health Crisis Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Global Mental Health Crisis Dashboard")
st.caption("Dataset: Global Mental Health Crisis Index 2026 (Kaggle)")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    # Đường dẫn tương đối tính từ file app.py
    data_file = os.path.join("data", "Global_Mental_Health_Crisis_Index_2026.csv")
    
    if not os.path.exists(data_file):
        st.error(f"Không tìm thấy file tại {data_file}")
        return None
        
    df = pd.read_csv(data_file)

    # Encode categorical column (Làm sạch dữ liệu ngay khi load)
    unique_labels = df["social_media_mental_health_risk"].unique()
    risk_mapping = {label: i for i, label in enumerate(unique_labels)}
    df["social_media_mh_risk_encoded"] = df["social_media_mental_health_risk"].map(risk_mapping)
    
    return df

with st.spinner("Đang tải dữ liệu..."):
    df = load_data()

st.success(f"✅ Đã tải {len(df):,} quốc gia")

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Bộ lọc")
    income_options = sorted(df["income_group_code"].dropna().unique())
    selected_income = st.multiselect(
        "Nhóm thu nhập",
        options=income_options,
        default=income_options,
    )
    region_options = sorted(df["region"].dropna().unique())
    selected_region = st.multiselect(
        "Khu vực",
        options=region_options,
        default=region_options,
    )

filtered_df = df[
    df["income_group_code"].isin(selected_income) &
    df["region"].isin(selected_region)
]
st.caption(f"Hiển thị **{len(filtered_df)}** quốc gia sau khi lọc")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan",
    "🌍 Overall Maps",
    "🏛️ Governance & System",
    "💰 Socio-Economics",
    "📱 Digital Life",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 – Tổng quan nhanh
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.subheader("Thống kê nhanh")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số quốc gia", len(filtered_df))
    c2.metric("MH Crisis Index TB", f"{filtered_df['mh_crisis_index'].mean():.2f}")
    c3.metric("Treatment Gap TB (%)", f"{filtered_df['treatment_gap_pct'].mean():.1f}%")
    c4.metric("Suicide Rate TB (per 100k)", f"{filtered_df['suicide_rate_per100k'].mean():.2f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Phân phối MH Crisis Index")
        fig_hist = px.histogram(
            filtered_df, x="mh_crisis_index", nbins=30,
            color_discrete_sequence=["#636EFA"],
            labels={"mh_crisis_index": "MH Crisis Index"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.subheader("Correlation Heatmap")
        numeric_df = filtered_df.select_dtypes(include=["float64", "int64"])
        fig_heat, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax, linewidths=0.3)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig_heat)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Overall Maps
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🗺️ Mental Health Crisis Index – Bản đồ toàn cầu")
    fig_map = px.choropleth(
        filtered_df,
        locations="iso3", locationmode="ISO-3",
        color="mh_crisis_index", hover_name="country",
        title="Mental Health Crisis Index",
        projection="natural earth",
        color_continuous_scale="Reds",
    )
    fig_map.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("🗺️ Treatment Gap – Bản đồ toàn cầu")
    fig_treatment = px.choropleth(
        filtered_df,
        locations="iso3", locationmode="ISO-3",
        color="treatment_gap_pct", hover_name="country",
        title="Treatment Gap by Country (%)",
        projection="natural earth",
        color_continuous_scale="Reds",
    )
    fig_treatment.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_treatment, use_container_width=True)

    st.subheader("📦 Tỉ lệ tự tử theo khu vực")
    fig_box = px.box(
        filtered_df,
        x="region", y="suicide_rate_per100k",
        color="region", points="all",
        hover_name="country",
        title="Phân bổ tỉ lệ tự tử theo từng khu vực",
        labels={"suicide_rate_per100k": "Tỉ lệ tự tử (trên 100k dân)", "region": "Khu vực"},
    )
    fig_box.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Treatment Gap vs Internet Penetration & Crisis Index")
    fig_scatter_overview = px.scatter(
        filtered_df,
        x="mh_crisis_index", y="treatment_gap_pct",
        hover_name="country",
        color="income_group_code",
        title="Crisis Index vs Treatment Gap",
        labels={"mh_crisis_index": "MH Crisis Index", "treatment_gap_pct": "Treatment Gap (%)"},
    )
    st.plotly_chart(fig_scatter_overview, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Governance & System
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🗺️ Mental Health System Scores")
    fig1_1 = px.choropleth(
        filtered_df,
        locations="iso3", locationmode="ISO-3",
        color="mh_system_score", hover_name="country",
        title="Global Mental Health System Scores by Country",
        projection="natural earth",
        color_continuous_scale="Blues",
    )
    fig1_1.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig1_1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Budget vs Crisis Index")
        fig1_2 = px.scatter(
            filtered_df,
            y="mh_budget_pct_health", x="mh_crisis_index",
            color="income_group_code", hover_name="country",
            trendline="ols", trendline_scope="overall",
            title="Mental Health Budget vs Crisis Index",
        )
        st.plotly_chart(fig1_2, use_container_width=True)

    with col2:
        st.subheader("Investment Gap vs Treatment Gap")
        fig3_5 = px.scatter(
            filtered_df,
            x="treatment_gap_pct", y="mh_investment_gap",
            color="income_group_code", hover_name="country",
            trendline="ols",
            title="Investment Gap vs Treatment Gap",
        )
        st.plotly_chart(fig3_5, use_container_width=True)

    st.subheader("Treatment Gap theo Income Group")
    fig1_3 = px.bar(
        filtered_df.sort_values("treatment_gap_pct"),
        x="country", y="treatment_gap_pct",
        color="income_group_code",
        title="Mental Health Treatment Gap Across Different Income Groups",
        labels={"treatment_gap_pct": "Treatment Gap (%)", "country": "Quốc gia"},
    )
    fig1_3.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig1_3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Socio-Economics & Human Resources
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bác sĩ tâm thần / 100k dân theo Income Group")
        fig2_1 = px.box(
            filtered_df,
            x="income_group_code", y="psychiatrists_per100k",
            color="income_group_code", points="all", hover_name="country",
            title="Bất bình đẳng nguồn lực bác sĩ theo nhóm thu nhập",
        )
        st.plotly_chart(fig2_1, use_container_width=True)

    with col2:
        st.subheader("GDP vs MH Crisis Index")
        fig2_2 = px.scatter(
            filtered_df,
            x="gdp_per_capita_usd", y="mh_crisis_index",
            size="population_millions", color="income_group_code",
            hover_name="country", log_x=True,
            title="GDP vs. Chỉ số Khủng hoảng (Size = Dân số)",
        )
        st.plotly_chart(fig2_2, use_container_width=True)

    st.subheader("Investment Gap theo Quốc gia")
    fig2_3 = px.bar(
        filtered_df.sort_values("mh_investment_gap"),
        y="mh_investment_gap", x="country",
        color="income_group_code",
        title="Mức đầu tư vào Mental Health theo Quốc gia",
        labels={"mh_investment_gap": "Investment Gap", "country": "Quốc gia"},
    )
    fig2_3.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig2_3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Digital Life & Well-being
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Mạng xã hội vs Lo âu")
    fig3_1 = px.scatter(
        filtered_df,
        x="social_media_hours_daily", y="anxiety_pct",
        color="social_media_mh_risk_encoded",
        size="population_millions",
        trendline="ols", hover_name="country",
        color_continuous_scale="Reds",
        title="Tác động của thời gian MXH đến Sự lo âu",
    )
    st.plotly_chart(fig3_1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Internet Penetration vs Treatment Gap")
        fig3_2 = px.scatter(
            filtered_df,
            x="internet_penetration_pct", y="treatment_gap_pct",
            color="income_group_code", hover_name="country",
            trendline="ols",
            title="Internet Penetration vs Treatment Gap",
        )
        st.plotly_chart(fig3_2, use_container_width=True)

    with col2:
        st.subheader("Internet Penetration vs Crisis Index")
        fig3_3 = px.scatter(
            filtered_df,
            x="internet_penetration_pct", y="mh_crisis_index",
            color="income_group_code", hover_name="country",
            trendline="ols",
            title="Internet Penetration vs Crisis Index",
        )
        st.plotly_chart(fig3_3, use_container_width=True)

    st.subheader("Internet Penetration vs Investment Gap")
    fig3_4 = px.scatter(
        filtered_df,
        x="internet_penetration_pct", y="mh_investment_gap",
        color="income_group_code", hover_name="country",
        trendline="ols",
        title="Internet Penetration vs Investment Gap",
    )
    st.plotly_chart(fig3_4, use_container_width=True)