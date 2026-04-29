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
    page_icon="✨",
    layout="wide",
)

st.title("✨ Global Mental Health Crisis Dashboard")
st.caption("Dataset: Global Mental Health Crisis Index 2026 (Kaggle)")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    # Đường dẫn tương đối tính từ file app.py
    data_file = os.path.join("data", "Global_Mental_Health_Crisis_Index_2026.csv")
    
    if not os.path.exists(data_file):
        st.error(f"Cannot find resources at: {data_file}")
        return None
        
    df = pd.read_csv(data_file)

    # Encode categorical column (Làm sạch dữ liệu ngay khi load)
    unique_labels = df["social_media_mental_health_risk"].unique()
    risk_mapping = {label: i for i, label in enumerate(unique_labels)}
    df["social_media_mh_risk_encoded"] = df["social_media_mental_health_risk"].map(risk_mapping)
    
    return df

with st.spinner("..."):
    df = load_data()

st.success(f"Loaded {len(df):,} nations!!!")

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 ")
    income_options = sorted(df["income_group_code"].dropna().unique())
    selected_income = st.multiselect(
        "Income group",
        options=income_options,
        default=income_options,
    )
    region_options = sorted(df["region"].dropna().unique())
    selected_region = st.multiselect(
        "Region",
        options=region_options,
        default=region_options,
    )

filtered_df = df[
    df["income_group_code"].isin(selected_income) &
    df["region"].isin(selected_region)
]
st.caption(f"**{len(filtered_df)}** nations filtered")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA Dataset",
    "🌍 Overall Maps",
    "🏛️ Governance & System",
    "💰 Socio-Economics",
    "📱 Digital Life",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 – Tổng quan nhanh
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.subheader("Quick statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of nation:", len(filtered_df))
    c2.metric("MH Crisis Index average", f"{filtered_df['mh_crisis_index'].mean():.2f}")
    c3.metric("Treatment Gap average (%)", f"{filtered_df['treatment_gap_pct'].mean():.1f}%")
    c4.metric("Suicide Rate average (per 100k)", f"{filtered_df['suicide_rate_per100k'].mean():.2f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("MH Crisis Index's distribution")
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
    # Chart 1 
    st.subheader("🗺️ Mental Health Crisis Index – Global map")
    fig_map = px.choropleth(
        filtered_df,
        locations="iso3", locationmode="ISO-3",
        color="mh_crisis_index", hover_name="country",
        title="Mental Health Crisis Index",
        projection="natural earth",
        color_continuous_scale="Reds",
    )
    fig_map.update_layout(
        height=500, 
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)', # Làm trong suốt phần nền ngoài
        plot_bgcolor='rgba(0,0,0,0)',  # Làm trong suốt phần nền biểu đồ
        font_color="white"             # Chỉnh chữ tiêu đề sang trắng
    )
    fig_map.update_geos(
        bgcolor='rgba(0,0,0,0)',       # Làm trong suốt nền của bản đồ thế giới
        showocean=False,               # Tắt màu đại dương để nó lấy màu nền Streamlit
        showlakes=False,               # Tắt màu hồ
        showcoastlines=True,
        coastlinecolor="Gray"          # Thêm đường viền lục địa cho dễ nhìn
    )
    st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")
    st.info("Global Mental Health Trends: The map highlights a critical "
            +"concentration of mental health crises in South American and African regions."
            +" In contrast, Northern Hemisphere countries generally maintain lower index scores, "
            +"indicating a strong correlation between geographic location and reported mental health "
            +"stability.")

    # Chart 2 
    st.subheader("🗺️ Treatment Gap – Global map")
    fig_treatment = px.choropleth(
        filtered_df,
        locations="iso3", locationmode="ISO-3",
        color="treatment_gap_pct", hover_name="country",
        title="Treatment Gap by Country (%)",
        projection="natural earth",
        color_continuous_scale="Reds",
    )
    fig_treatment.update_layout(
        height=500, 
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)', # Làm trong suốt phần nền ngoài
        plot_bgcolor='rgba(0,0,0,0)',  # Làm trong suốt phần nền biểu đồ
        font_color="white"             # Chỉnh chữ tiêu đề sang trắng
    )
    fig_treatment.update_geos(
        bgcolor='rgba(0,0,0,0)',       # Làm trong suốt nền của bản đồ thế giới
        showocean=False,               # Tắt màu đại dương để nó lấy màu nền Streamlit
        showlakes=False,               # Tắt màu hồ
        showcoastlines=True,
        coastlinecolor="Gray"          # Thêm đường viền lục địa cho dễ nhìn
    )
    st.plotly_chart(fig_treatment, use_container_width=True, theme="streamlit")
    st.warning("This map highlights a staggering Treatment Gap in mental health care "+
               "across the Global South. While developed regions like North America and "+
               "Australia maintain a lower gap (under 20%), many nations in Africa, Southeast "+
               "Asia, and South America face a gap exceeding 80%. This suggests that in these "+
               "high-intensity areas, the vast majority of individuals with mental health conditions "+
               "do not receive the necessary clinical support.")

    # Chart 3 
    st.subheader("📦 Suicide rates by region")
    fig_box = px.box(
        filtered_df,
        x="region", y="suicide_rate_per100k",
        color="region", points="all",
        hover_name="country",
        title="Distribution of suicide rates by region",
        labels={"suicide_rate_per100k": "Suicide rate (per 100,000 people)", "region": ""},
    )
    fig_box.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")
    st.info("Key Insight: Regional suicide rates show distinct patterns of distribution. "+
            "Europe and the Americas display more concentrated and generally higher median rates "+
            "compared to the Eastern Mediterranean, which maintains the lowest overall distribution. "+
            "The presence of significant outliers in Africa and Europe suggests that while regional "+
            "averages provide a baseline, specific countries within these areas face disproportionately "+
            "high mental health challenges.")

    # Chart 4 
    st.subheader("Treatment Gap vs Internet Penetration & Crisis Index")
    fig_scatter_overview = px.scatter(
        filtered_df,
        x="mh_crisis_index", y="treatment_gap_pct",
        hover_name="country",
        color="income_group_code",
        title="Crisis Index vs Treatment Gap",
        labels={"mh_crisis_index": "Mental Health Crisis Index", "treatment_gap_pct": "Treatment Gap (%)"},
    )
    st.plotly_chart(fig_scatter_overview, use_container_width=True, theme="streamlit")
    st.info("Key Insight: There is a visible clustering based on income groups. "+
            "Lower income group codes (represented by darker blue) tend to occupy "+
            "the top-right quadrant, indicating both a higher mental health crisis "+
            "and a wider treatment gap. Conversely, higher-income nations (lighter blue) "+
            "cluster in the bottom-left, showing more resilient mental health infrastructures "+
            "with lower gaps and lower crisis levels.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Governance & System
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    # Chart 1 
    st.subheader("🗺️ Mental Health System Scores")
    fig1_1 = px.choropleth(
        filtered_df,
        locations="iso3", locationmode="ISO-3",
        color="mh_system_score", hover_name="country",
        title="Global Mental Health System Scores by Country",
        projection="natural earth",
        color_continuous_scale="Blues",
    )
    fig1_1.update_layout(
        height=500, 
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)', # Làm trong suốt phần nền ngoài
        plot_bgcolor='rgba(0,0,0,0)',  # Làm trong suốt phần nền biểu đồ
        font_color="white"             # Chỉnh chữ tiêu đề sang trắng
    )
    fig1_1.update_geos(
        bgcolor='rgba(0,0,0,0)',       # Làm trong suốt nền của bản đồ thế giới
        showocean=False,               # Tắt màu đại dương để nó lấy màu nền Streamlit
        showlakes=False,               # Tắt màu hồ
        showcoastlines=True,
        coastlinecolor="Gray"          # Thêm đường viền lục địa cho dễ nhìn
    )
    st.plotly_chart(fig1_1, use_container_width=True, theme="streamlit")
    st.info("Key Insight: Mental health system strength is heavily correlated with "+
            "regional economic development. While Europe and Australia exhibit deep "+
            "blue shades (indicating mature, well-funded systems), regions across Africa "+
            "and parts of Central Asia remain in the lightest blue range (under 20). "+
            "This disparity underscores the urgent need for structural investment in "+
            "mental health resources within emerging economies to move beyond basic crisis "+
            "management.")

    # Chart 2, 3 
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
    st.info("These charts collectively highlight a profound socio-economic divide: "+
            "there is a clear inverse correlation where higher budget allocations and "+
            "narrower investment gaps are foundational to lowering crisis levels and "+
            "improving treatment accessibility. While high-income nations (light blue) "+
            "successfully stabilize their mental health landscapes through robust funding "+
            "(often above 6-8% of health budgets), lower-income regions (dark blue) remain "+
            "trapped in a systemic cycle where underinvestment directly exacerbates both the "+
            "intensity of the mental health crisis and the staggering gap in clinical care.")

    # Chart 4: 
    st.subheader("Treatment Gap theo Income Group")
    fig1_3 = px.bar(
        filtered_df.sort_values("treatment_gap_pct"),
        x="country", y="treatment_gap_pct",
        color="income_group_code",
        title="Mental Health Treatment Gap Across Different Income Groups",
        labels={"treatment_gap_pct": "Treatment Gap (%)", "country": "Nations"},
    )
    fig1_3.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig1_3, use_container_width=True)
    st.info("The bar chart reveals a steep escalation in the Treatment Gap as we move "+
            "from high-income to low-income nations. Developed countries like Switzerland "+
            "and the US maintain the lowest gaps (under 20%), whereas low-income economies "+
            "(dark blue bars) such as Mozambique and Tanzania face critical gaps nearing 95%. "+
            "This visual progression underscores that mental health care remains a luxury of "+
            "wealth, with the vast majority of populations in developing nations left entirely "+
            "without professional support.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Socio-Economics & Human Resources
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    #Chart 1, 2 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Psychiatrist / per 100,000 people, according to Income Group")
        fig2_1 = px.box(
            filtered_df,
            x="income_group_code", y="psychiatrists_per100k",
            color="income_group_code", points="all", hover_name="country",
            title="Inequality in doctor resources by income group",
        )
        st.plotly_chart(fig2_1, use_container_width=True)

    with col2:
        st.subheader("GDP vs Mental Health Crisis Index")
        fig2_2 = px.scatter(
            filtered_df,
            x="gdp_per_capita_usd", y="mh_crisis_index",
            size="population_millions", color="income_group_code",
            hover_name="country", log_x=True,
            title="GDP vs. Crisis Index (Size = Population)",
        )
        st.plotly_chart(fig2_2, use_container_width=True)
    st.info("The data reveals a stark resource-wealth paradox: while high-income "+
            "nations (Group 4) possess a significant concentration of specialized "+
            "human capital—averaging over 15 psychiatrists per 100,000 people—low—income "+
            "regions (Group 1 & 2) effectively have near-zero specialist "+
            "availability. This lack of personnel coincides with a clear negative "+
            "correlation between GDP and mental health stability; as GDP per capita "+
            "drops, the Crisis Index climbs toward critical levels (80+). Collectively, "+
            "these insights underscore a global emergency where the populations with the"+
            " highest mental health needs are precisely those with the least financial and"+
            " professional resources to address them.")

    #Chart 3:
    st.subheader("Investment Gap by Country")
    fig2_3 = px.bar(
        filtered_df.sort_values("mh_investment_gap"),
        y="mh_investment_gap", x="country",
        color="income_group_code",
        title="Investment in Mental Health by Country",
        labels={"mh_investment_gap": "Investment Gap", "country": "Country"},
    )
    fig2_3.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig2_3, use_container_width=True)
    st.info("The bar chart illustrates a significant disparity in mental health"+
            " investment gaps across nations, often mirroring economic capacity."+
            " Interestingly, high-income nations like Germany and the UAE (represented "+
            "by light blue bars) show some of the widest investment gaps, potentially "+
            "due to the high cost of advanced psychiatric infrastructure and rising "+
            "service demands. Conversely, lower-income nations (dark blue) show smaller "+
            "numerical gaps, which likely reflects minimal baseline targets rather than "+
            "sufficient funding. This trend suggests that as nations develop, the financial "+
            "requirements to bridge the gap between existing mental health services and ideal "+
            "clinical standards become increasingly complex and expensive.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Digital Life & Well-being
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    #Chart 1: 
    st.subheader("Social media vs. anxiety")
    fig3_1 = px.scatter(
        filtered_df,
        x="social_media_hours_daily", y="anxiety_pct",
        color="social_media_mh_risk_encoded",
        size="population_millions",
        trendline="ols", hover_name="country",
        color_continuous_scale="Reds",
        title="The impact of social media time on anxiety",
    )
    st.plotly_chart(fig3_1, use_container_width=True)
    st.info("The visualization reveals a positive correlation between daily social media "+
            "consumption and anxiety levels. As daily usage climbs toward 3 to 5 hours, there "+
            "is a measurable upward trend in reported anxiety percentages, moving from a "+
            "baseline of 4% to over 8%. Furthermore, the color encoding indicates that higher"+
            " usage is frequently associated with an increased mental health risk score (dark "+
            "red clusters). This suggests that while moderate use (under 2 hours) aligns with "+
            "lower anxiety, excessive screen time serves as a significant lifestyle stressor "+
            "that compounds psychological risk.")

    #Chart 2,3: 
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
    st.info("The analysis reveals a strong negative correlation between Internet Penetration "+
            "and both the Treatment Gap and the Crisis Index. As digital connectivity increases "+
            "beyond 60-80%, there is a sharp decline in the Treatment Gap (falling from ~90% to"+
            " below 40%) and a corresponding stabilization of the Crisis Index. This suggests "+
            "that high internet penetration—primarily seen in high-income nations (light blue)—acts"+
            " as a proxy for better healthcare infrastructure, improved mental health literacy, and"+
            " the availability of digital intervention tools. Conversely, 'digitally isolated' "+
            "regions face a double burden of high crisis levels and nearly insurmountable barriers"+
            " to treatment, highlighting the digital divide as a key factor in global mental health"+
            " inequality.")
        
    #Chart 4: 
    st.subheader("Internet Penetration vs Investment Gap")
    fig3_4 = px.scatter(
        filtered_df,
        x="internet_penetration_pct", y="mh_investment_gap",
        color="income_group_code", hover_name="country",
        trendline="ols",
        title="Internet Penetration vs Investment Gap",
    )
    st.plotly_chart(fig3_4, use_container_width=True)
    st.info("The scatter plot reveals a positive correlation between Internet Penetration "+
            "and the Mental Health Investment Gap. At first glance, it appears paradoxical "+
            "that digitally advanced nations (light blue) face wider investment gaps; however, "+
            "this trend underscores the rising costs of sophisticated care. As internet access "+
            "expands, public awareness and the demand for high-quality, tech-integrated mental "+
            "health services increase, creating a larger financial target to reach 'ideal' "+
            "system standards. Conversely, nations with low internet penetration (dark blue)"+
            " show smaller investment gaps, likely due to lower baseline service expectations"+
            " rather than adequate funding, highlighting how digital maturity shifts the economic"+
            " scale of mental healthcare needs.")