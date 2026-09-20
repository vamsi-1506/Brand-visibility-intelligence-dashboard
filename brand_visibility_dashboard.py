import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Brand Visibility Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Configuration
# ============================================================

DB_PATH = Path("products.db")
TABLE_NAME = "products_data"
CSV_FALLBACK = Path("complete_dataset_with_all_features.csv")


# ============================================================
# Database helpers
# ============================================================

@st.cache_resource
def get_connection():
    """Create one SQLite connection for the Streamlit session."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def get_columns():
    """Return database table columns."""
    conn = get_connection()

    rows = conn.execute(
        f"PRAGMA table_info({TABLE_NAME})"
    ).fetchall()

    return [row[1] for row in rows]


def initialize_database():
    """
    Use the existing products.db if available.
    If it is not available, create it from the CSV.
    """

    if DB_PATH.exists():
        return

    if CSV_FALLBACK.exists():

        df = pd.read_csv(CSV_FALLBACK)

        conn = sqlite3.connect(DB_PATH)

        df.to_sql(
            TABLE_NAME,
            conn,
            if_exists="replace",
            index=False
        )

        conn.close()

        get_connection.clear()
        get_columns.clear()

    else:

        st.error(
            "products.db was not found, and "
            "complete_dataset_with_all_features.csv was also not found."
        )

        st.stop()


initialize_database()

COLUMNS = get_columns()


# ============================================================
# SQL utilities
# ============================================================

def quote_identifier(identifier):
    """Safely quote a SQLite identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def column_exists(column_name):
    return column_name in COLUMNS


def get_distinct_values(column_name):

    if not column_exists(column_name):
        return []

    conn = get_connection()

    query = f"""
        SELECT DISTINCT {quote_identifier(column_name)}
        FROM {quote_identifier(TABLE_NAME)}
        WHERE {quote_identifier(column_name)} IS NOT NULL
          AND TRIM(CAST({quote_identifier(column_name)} AS TEXT)) <> ''
        ORDER BY {quote_identifier(column_name)}
    """

    rows = conn.execute(query).fetchall()

    return [row[0] for row in rows]


def get_numeric_range(column_name):

    if not column_exists(column_name):
        return 0.0, 0.0

    conn = get_connection()

    query = f"""
        SELECT MIN({quote_identifier(column_name)}),
               MAX({quote_identifier(column_name)})
        FROM {quote_identifier(TABLE_NAME)}
        WHERE {quote_identifier(column_name)} IS NOT NULL
    """

    result = conn.execute(query).fetchone()

    if (
        not result
        or result[0] is None
        or result[1] is None
    ):
        return 0.0, 0.0

    return float(result[0]), float(result[1])


def build_where_clause(
    brands=None,
    platforms=None,
    keywords=None,
    price_range=None,
    rating_range=None,
    search_text=None,
):

    conditions = []
    params = []

    if brands and column_exists("brand"):

        placeholders = ",".join(["?"] * len(brands))

        conditions.append(
            f'{quote_identifier("brand")} IN ({placeholders})'
        )

        params.extend(brands)

    if platforms and column_exists("platform"):

        placeholders = ",".join(["?"] * len(platforms))

        conditions.append(
            f'{quote_identifier("platform")} IN ({placeholders})'
        )

        params.extend(platforms)

    if keywords and column_exists("keyword"):

        placeholders = ",".join(["?"] * len(keywords))

        conditions.append(
            f'{quote_identifier("keyword")} IN ({placeholders})'
        )

        params.extend(keywords)

    if price_range and column_exists("price"):

        conditions.append(
            f'{quote_identifier("price")} BETWEEN ? AND ?'
        )

        params.extend([
            price_range[0],
            price_range[1]
        ])

    if rating_range and column_exists("rating"):

        conditions.append(
            f'{quote_identifier("rating")} BETWEEN ? AND ?'
        )

        params.extend([
            rating_range[0],
            rating_range[1]
        ])

    if search_text and column_exists("title"):

        conditions.append(
            f'LOWER(CAST({quote_identifier("title")} AS TEXT)) LIKE ?'
        )

        params.append(
            f"%{search_text.lower()}%"
        )

    where_sql = ""

    if conditions:

        where_sql = " WHERE " + " AND ".join(conditions)

    return where_sql, params


# ============================================================
# UPDATED QUERY FUNCTION
# ============================================================

@st.cache_data(show_spinner=False)
def run_filtered_query(
    select_sql="*",
    brands=None,
    platforms=None,
    keywords=None,
    price_range=None,
    rating_range=None,
    search_text=None,
    extra_where="",
    extra_params=None,
    limit=None,
    order_by=None,
    group_by=None,
):

    """
    Run a SELECT query with global sidebar filters.

    group_by is used for aggregation queries such as:
        COUNT(*) by brand
        COUNT(*) by platform
        AVG(rating) by brand
    """

    where_sql, params = build_where_clause(
        brands=brands,
        platforms=platforms,
        keywords=keywords,
        price_range=price_range,
        rating_range=rating_range,
        search_text=search_text,
    )

    query = (
        f"SELECT {select_sql} "
        f"FROM {quote_identifier(TABLE_NAME)} "
        f"{where_sql}"
    )

    if extra_where:

        query += (
            " AND " if where_sql else " WHERE "
        ) + extra_where

    params = params + (extra_params or [])

    # IMPORTANT:
    # GROUP BY must come before ORDER BY and LIMIT
    if group_by:
        query += f" GROUP BY {group_by}"

    if order_by:
        query += f" ORDER BY {order_by}"

    if limit:
        query += f" LIMIT {int(limit)}"

    conn = get_connection()

    return pd.read_sql_query(
        query,
        conn,
        params=params
    )


# ============================================================
# Cached sidebar metadata
# ============================================================

@st.cache_data
def sidebar_metadata():

    return {
        "brands": get_distinct_values("brand"),
        "platforms": get_distinct_values("platform"),
        "keywords": get_distinct_values("keyword"),
        "price_range": get_numeric_range("price"),
        "rating_range": get_numeric_range("rating"),
    }


META = sidebar_metadata()


# ============================================================
# Sidebar
# ============================================================

def reset_filters():

    st.session_state["filter_brands"] = []
    st.session_state["filter_platforms"] = []
    st.session_state["filter_keywords"] = []
    st.session_state["filter_price"] = (
        float(META["price_range"][0]),
        float(META["price_range"][1]),
    )
    st.session_state["filter_rating"] = (
        float(META["rating_range"][0]),
        float(META["rating_range"][1]),
    )

st.sidebar.title("🎛️ Dashboard Filters")

st.sidebar.caption(
    "All filters are applied through SQL to every dashboard page."
)


selected_brands = st.sidebar.multiselect(
    "Brand",
    options=META["brands"],
    default=[],
    key="filter_brands",
)


selected_platforms = st.sidebar.multiselect(
    "Platform",
    options=META["platforms"],
    default=[],
    key="filter_platforms",
)


selected_keywords = st.sidebar.multiselect(
    "Keyword",
    options=META["keywords"],
    default=[],
    key="filter_keywords",
)


min_price, max_price = META["price_range"]


if min_price < max_price:

    selected_price = st.sidebar.slider(
        "Price Range",
        min_value=float(min_price),
        max_value=float(max_price),
        value=(
            float(min_price),
            float(max_price)
        ),
        key="filter_price",
    )

else:

    selected_price = (
        float(min_price),
        float(max_price)
    )


min_rating, max_rating = META["rating_range"]


if min_rating < max_rating:

    selected_rating = st.sidebar.slider(
        "Rating Range",
        min_value=float(min_rating),
        max_value=float(max_rating),
        value=(
            float(min_rating),
            float(max_rating)
        ),
        step=0.1,
        key="filter_rating",
    )

else:

    selected_rating = (
        float(min_rating),
        float(max_rating)
    )


FILTERS = {
    "brands": selected_brands,
    "platforms": selected_platforms,
    "keywords": selected_keywords,
    "price_range": selected_price,
    "rating_range": selected_rating,
}


if st.sidebar.button(
    "🔄 Reset Filters",
    use_container_width=True,
    on_click=reset_filters,
):

    st.query_params.clear()


# ============================================================
# Common helpers
# ============================================================

def money(value):

    if pd.isna(value):
        return "₹0"

    return f"₹{value:,.2f}"


def number(value):

    if pd.isna(value):
        return "0"

    return f"{value:,.0f}"


def metric_row(metrics):

    cols = st.columns(len(metrics))

    for col, (label, value, help_text) in zip(
        cols,
        metrics
    ):

        col.metric(
            label,
            value,
            help=help_text
        )


def filtered_count():

    df = run_filtered_query(
        select_sql="COUNT(*) AS total_products",
        **FILTERS,
    )

    return int(
        df.iloc[0]["total_products"]
    )


def filtered_dataframe(limit=None):

    return run_filtered_query(
        select_sql="*",
        **FILTERS,
        limit=limit,
    )


def show_active_filters():

    active = []

    if selected_brands:
        active.append(
            f"Brands: {len(selected_brands)}"
        )

    if selected_platforms:
        active.append(
            f"Platforms: {len(selected_platforms)}"
        )

    if selected_keywords:
        active.append(
            f"Keywords: {len(selected_keywords)}"
        )

    if (
        selected_price[0] != min_price
        or selected_price[1] != max_price
    ):

        active.append(
            f"Price: {money(selected_price[0])} – "
            f"{money(selected_price[1])}"
        )

    if (
        selected_rating[0] != min_rating
        or selected_rating[1] != max_rating
    ):

        active.append(
            f"Rating: {selected_rating[0]:.1f} – "
            f"{selected_rating[1]:.1f}"
        )

    if active:

        st.info(" | ".join(active))

    else:

        st.caption(
            "No sidebar filters selected — "
            "showing the full dataset."
        )


# ============================================================
# Header
# ============================================================

st.title(
    "📊 Brand Visibility Intelligence Dashboard"
)

st.caption(
    "SQL-connected Streamlit dashboard using the same "
    "Seaborn/Matplotlib visualization approach used in "
    "Project2_streamlit_visualizations(1).ipynb."
)


page = st.radio(
    "Navigate",
    [
        "Overview",
        "Brand Insights",
        "Pricing Analysis",
        "Platform Analysis",
        "Visibility & Ranking",
        "Product Explorer",
    ],
    horizontal=True,
)


show_active_filters()


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "Overview":

    st.header("1. Overview")

    kpis = run_filtered_query(
        select_sql="""
            COUNT(*) AS total_products,
            AVG(price) AS avg_price,
            AVG(rating) AS avg_rating,
            SUM(reviews) AS total_reviews
        """,
        **FILTERS,
    ).iloc[0]


    metric_row([
        (
            "Total Products",
            number(kpis["total_products"]),
            "Number of products after filters"
        ),

        (
            "Avg Price",
            money(kpis["avg_price"]),
            "Average product price after filters"
        ),

        (
            "Avg Rating",
            f'{kpis["avg_rating"]:.2f}'
            if pd.notna(kpis["avg_rating"])
            else "0.00",
            "Average rating after filters"
        ),

        (
            "Total Reviews",
            number(kpis["total_reviews"]),
            "Sum of reviews after filters"
        ),
    ])


    st.divider()


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Price distribution
    # --------------------------------------------------------

    with col1:

        st.subheader("OV-01 | Price Distribution")
        
        df_price = run_filtered_query(
            select_sql="price",
            **FILTERS,
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.histplot(
            df_price["price"],
            bins=50,
            log_scale=True,
            kde=True,
            color="skyblue",
            ax=ax,
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, position: f"{value:,.0f}"
            )
        )

        ax.set_title(
            "Distribution of Product Prices from Database"
        )

        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Number of Products")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # KEYWORD BAR CHART
    # --------------------------------------------------------

    with col2:

        st.subheader("OV-02 | Products per Keyword")

        df_keyword = run_filtered_query(
            select_sql="keyword, COUNT(*) AS product_count",
            **FILTERS,
            extra_where="keyword IS NOT NULL",
            group_by="keyword",                 # FIX
            order_by="product_count DESC",
        )

        keyword_plot = (
            df_keyword
            .head(20)
            .copy()
        )


        fig, ax = plt.subplots(
            figsize=(12, 7)
        )


        sns.barplot(
            x="product_count",
            y="keyword",
            data=keyword_plot,
            hue="keyword",
            ax=ax,
        )

        ax.legend(
            title="Keyword",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )


        ax.set_title(
            "Top 20 Keywords by Number of Products from Database"
        )

        ax.set_xlabel("Number of Products")
        ax.set_ylabel("Keyword")


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # PLATFORM SHARE
    # --------------------------------------------------------

    st.subheader("OV-03 | Platform Share")

    df_platform = run_filtered_query(
        select_sql="platform, COUNT(*) AS product_count",
        **FILTERS,
        extra_where="platform IS NOT NULL",
        group_by="platform",                    # FIX
        order_by="product_count DESC",
    )


    platform_share = (
        df_platform
        .set_index("platform")["product_count"]
    )


    platform_share_pct = (
        platform_share
        / platform_share.sum()
        * 100
    )


    major_platforms = platform_share_pct[
        platform_share_pct >= 2
    ]


    others_share = platform_share_pct[
        platform_share_pct < 2
    ].sum()


    plot_data = major_platforms.copy()


    if others_share > 0:
        plot_data["Others"] = others_share


    plot_data = plot_data.sort_values(
        ascending=False
    )


    fig, ax = plt.subplots(
        figsize=(10, 10)
    )


    ax.pie(
        plot_data,
        labels=plot_data.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette(
            "pastel",
            len(plot_data)
        ),
    )


    ax.set_title(
        "Platform Share of Products"
    )

    ax.axis("equal")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 2. BRAND INSIGHTS
# ============================================================

elif page == "Brand Insights":

    st.header("2. Brand Insights")


    brand_filter = (
        "brand IS NOT NULL "
        "AND TRIM(brand) <> '' "
        "AND brand <> 'Unknown'"
    )


    brand_kpis = run_filtered_query(
        select_sql="""
            COUNT(*) AS product_count,
            AVG(popularity_score) AS avg_visibility
        """,
        **FILTERS,
        extra_where=brand_filter,
    ).iloc[0]


    # --------------------------------------------------------
    # TOP BRAND
    # --------------------------------------------------------

    top_brand_df = run_filtered_query(
        select_sql="brand, COUNT(*) AS product_count",
        **FILTERS,
        extra_where=brand_filter,
        group_by="brand",                       # FIX
        order_by="product_count DESC",
        limit=1,
    )


    top_brand = (
        top_brand_df.iloc[0]["brand"]
        if not top_brand_df.empty
        else "N/A"
    )


    metric_row([
        (
            "Top Brand",
            str(top_brand),
            "Brand with the highest filtered product count"
        ),

        (
            "Avg Visibility Score",
            f'{brand_kpis["avg_visibility"]:.2f}'
            if pd.notna(brand_kpis["avg_visibility"])
            else "0.00",
            "Average popularity_score after filters"
        ),
    ])


    st.divider()


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # BRAND COUNT BAR CHART
    # --------------------------------------------------------

    with col1:

        st.subheader("BI-01 | Brand vs Product Count")

        brand_counts = run_filtered_query(
            select_sql="brand, COUNT(*) AS product_count",
            **FILTERS,
            extra_where=brand_filter,
            group_by="brand",                   # FIX
            order_by="product_count DESC",
            limit=20,
        )


        brand_plot = (
            brand_counts
            .sort_values(
                "product_count",
                ascending=False
            )
            .head(20)
        )


        fig, ax = plt.subplots(
            figsize=(12, 6)
        )


        sns.barplot(
            x="product_count",
            y="brand",
            data=brand_plot,
            hue="brand",
            ax=ax,
        )

        ax.legend(
            title="Brand",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )


        ax.set_title(
            "Top 20 Brands by Product Count"
        )

        ax.set_xlabel(
            "Number of Products"
        )

        ax.set_ylabel("Brand")


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # BRAND RATING BAR CHART
    # --------------------------------------------------------

    with col2:

        st.subheader("BI-02 | Brand vs Avg Rating")

        brand_ratings = run_filtered_query(
            select_sql="""
                brand,
                AVG(rating) AS avg_rating,
                COUNT(*) AS product_count
            """,
            **FILTERS,
            extra_where=(
                f"{brand_filter} "
                "AND rating IS NOT NULL"
            ),
            group_by="brand",                   # FIX
            order_by="product_count DESC",
        )


        brand_ratings = brand_ratings[
            brand_ratings["product_count"] >= 5
        ]


        brand_ratings = (
            brand_ratings
            .sort_values(
                "avg_rating",
                ascending=False
            )
            .head(20)
            .sort_values("avg_rating")
        )


        fig, ax = plt.subplots(
            figsize=(12, 6)
        )


        sns.barplot(
            x="avg_rating",
            y="brand",
            data=brand_ratings,
            hue="brand",
            ax=ax,
        )

        ax.legend(
            title="Brand",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )


        ax.set_title(
            "Top 20 Brands by Average Rating (Min 5 products)"
        )

        ax.set_xlabel(
            "Average Rating"
        )

        ax.set_ylabel("Brand")

        ax.set_xlim(1, 5)


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # TOP 10 VISIBILITY
    # --------------------------------------------------------

    st.subheader("BI-03 | Top Brands in Top 10 Positions")


    if column_exists("popularity_score"):

        top10 = run_filtered_query(
            select_sql="""
                brand,
                title,
                popularity_score
            """,
            **FILTERS,
            extra_where=(
                f"{brand_filter} "
                "AND popularity_score IS NOT NULL"
            ),
            order_by="popularity_score DESC",
            limit=10,
        )


        top10_brands = (
            top10
            .groupby("brand")
            .size()
            .reset_index(
                name="count_in_top_10"
            )
            .sort_values(
                "count_in_top_10",
                ascending=False
            )
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.barplot(
            x="count_in_top_10",
            y="brand",
            data=top10_brands,
            hue="brand",
            ax=ax,
        )

        ax.legend(
            title="Brand",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )


        ax.set_title(
            "Brand Occurrences in Top 10 Most Popular Products"
        )

        ax.set_xlabel("Occurrences")
        ax.set_ylabel("Brand")


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# 3. PRICING ANALYSIS
# ============================================================

elif page == "Pricing Analysis":

    st.header("3. Pricing Analysis")


    if column_exists("discount"):

        discounted_expression = (
            "100.0 * "
            "SUM(CASE WHEN discount > 0 THEN 1 ELSE 0 END) "
            "/ NULLIF(COUNT(*), 0)"
        )

    else:

        discounted_expression = "0.0"


    pricing_kpis = run_filtered_query(
        select_sql=f"""
            AVG(price) AS avg_price,
            MAX(price) AS max_price,
            {discounted_expression} AS pct_discounted
        """,
        **FILTERS,
    ).iloc[0]


    metric_row([
        (
            "Avg Price",
            money(pricing_kpis["avg_price"]),
            "Average filtered product price"
        ),

        (
            "Max Price",
            money(pricing_kpis["max_price"]),
            "Maximum filtered product price"
        ),

        (
            "% Discounted Products",
            f'{pricing_kpis["pct_discounted"]:.2f}%',
            "Percentage of filtered products with discount > 0"
        ),
    ])


    st.divider()


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # PRICE HISTOGRAM
    # --------------------------------------------------------

    with col1:

        st.subheader("PA-01 | Price Distribution")

        price_df = run_filtered_query(
            select_sql="price",
            **FILTERS,
        )


        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        sns.histplot(
            price_df["price"],
            bins=50,
            kde=True,
            color="teal",
            ax=ax,
        )


        ax.set_title(
            "Overall Distribution of Product Prices"
        )

        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Count")


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col2:

        st.subheader("PA-02 | Price vs Ranking")

        if column_exists("popularity_score"):

            price_rank = run_filtered_query(
                select_sql="""
                    price,
                    popularity_score
                """,
                **FILTERS,
                extra_where=(
                    "price IS NOT NULL "
                    "AND popularity_score IS NOT NULL"
                ),
            )


            fig, ax = plt.subplots(
                figsize=(10, 6)
            )


            sns.scatterplot(
                data=price_rank,
                x="price",
                y="popularity_score",
                alpha=0.5,
                color="coral",
                ax=ax,
            )


            ax.set_title(
                "Product Price vs. Popularity Score"
            )

            ax.set_xlabel("Price (₹)")
            ax.set_ylabel("Popularity Score")

            ax.set_xscale("log")


            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


    # --------------------------------------------------------
    # PRICE VS RATING
    # --------------------------------------------------------

    st.subheader("PA-03 | Price vs Rating")

    price_rating = run_filtered_query(
        select_sql="price, rating",
        **FILTERS,
        extra_where=(
            "price IS NOT NULL "
            "AND rating IS NOT NULL"
        ),
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    sns.scatterplot(
        data=price_rating,
        x="price",
        y="rating",
        alpha=0.5,
        color="purple",
        ax=ax,
    )


    ax.set_title(
        "Product Price vs. Customer Rating"
    )

    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Rating (1-5)")

    ax.set_xscale("log")
    ax.set_ylim(1, 5)


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 4. PLATFORM ANALYSIS
# ============================================================

elif page == "Platform Analysis":

    st.header("4. Platform Analysis")


    platform_kpis = run_filtered_query(
        select_sql="""
            COUNT(DISTINCT platform) AS total_platforms,
            AVG(rating) AS overall_avg_rating
        """,
        **FILTERS,
    ).iloc[0]


    # --------------------------------------------------------
    # BEST PLATFORM
    # --------------------------------------------------------

    best_platform_df = run_filtered_query(
        select_sql="""
            platform,
            AVG(rating) AS avg_rating,
            COUNT(*) AS product_count
        """,
        **FILTERS,
        extra_where=(
            "platform IS NOT NULL "
            "AND rating IS NOT NULL"
        ),
        group_by="platform",                   # FIX
        order_by="avg_rating DESC",
        limit=1,
    )


    best_platform = (
        best_platform_df.iloc[0]["platform"]
        if not best_platform_df.empty
        else "N/A"
    )


    metric_row([
        (
            "Total Platforms",
            number(
                platform_kpis["total_platforms"]
            ),
            "Distinct platforms after filters"
        ),

        (
            "Best Platform (Avg Rating)",
            str(best_platform),
            "Platform with highest filtered average rating"
        ),
    ])


    st.divider()


    # --------------------------------------------------------
    # PLATFORM STATISTICS
    # --------------------------------------------------------

    platform_stats = run_filtered_query(
        select_sql="""
            platform,
            COUNT(*) AS product_count,
            AVG(price) AS avg_price,
            AVG(rating) AS avg_rating
        """,
        **FILTERS,
        extra_where="platform IS NOT NULL",
        group_by="platform",                   # FIX
        order_by="product_count DESC",
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # PRODUCT COUNT BY PLATFORM
    # --------------------------------------------------------

    with col1:

        st.subheader("PL-01 | Platform vs Product Count")

        platform_plot = (
            platform_stats
            .head(20)
            .sort_values(
                "product_count",
                ascending=False
            )
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        sns.barplot(
            x="product_count",
            y="platform",
            data=platform_plot,
            hue="platform",
            ax=ax,
        )

        ax.legend(
            title="Platform",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )


        ax.set_title(
            "Product Count by Platform"
        )

        ax.set_xlabel(
            "Number of Products"
        )

        ax.set_ylabel("Platform")


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # AVERAGE PRICE BY PLATFORM
    # --------------------------------------------------------

    with col2:

        st.subheader("PL-02 | Platform vs Avg Price")

        price_data = (
            platform_stats
            .dropna(subset=["avg_price"])
            .head(20)
        )


        price_data = price_data.sort_values(
            "avg_price"
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        sns.barplot(
            x="avg_price",
            y="platform",
            data=price_data,
            hue="platform",
            ax=ax,
        )

        ax.legend(
            title="Platform",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )


        ax.set_title(
            "Average Product Price by Platform"
        )

        ax.set_xlabel(
            "Average Price (₹)"
        )

        ax.set_ylabel("Platform")


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # AVERAGE RATING BY PLATFORM
    # --------------------------------------------------------

    st.subheader("PL-03 | Platform vs Avg Rating")

    rating_data = (
        platform_stats
        .dropna(subset=["avg_rating"])
        .sort_values("avg_rating")
    )


    rating_plot = rating_data.head(20)


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.barplot(
        x="avg_rating",
        y="platform",
        data=rating_plot,
        hue="platform",
        ax=ax,
    )

    ax.legend(
        title="Platform",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )


    ax.set_title(
        "Average Customer Rating by Platform"
    )

    ax.set_xlabel(
        "Average Rating (1-5)"
    )

    ax.set_ylabel("Platform")

    ax.set_xlim(1, 5)


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 5. VISIBILITY & RANKING
# ============================================================

elif page == "Visibility & Ranking":

    st.header("5. Visibility & Ranking")


    if not column_exists("popularity_score"):

        st.warning(
            "The database does not contain the notebook's "
            "popularity_score column, so ranking/visibility "
            "charts cannot be displayed."
        )

        st.stop()


    visibility_kpis = run_filtered_query(
        select_sql="""
            AVG(rating) AS avg_rating,
            AVG(popularity_score) AS avg_visibility
        """,
        **FILTERS,
    ).iloc[0]


    metric_row([
        (
            "Avg Rating",
            f'{visibility_kpis["avg_rating"]:.2f}'
            if pd.notna(
                visibility_kpis["avg_rating"]
            )
            else "0.00",
            "Average filtered rating"
        ),

        (
            "Avg Visibility Score",
            f'{visibility_kpis["avg_visibility"]:.2f}'
            if pd.notna(
                visibility_kpis["avg_visibility"]
            )
            else "0.00",
            "Average popularity_score"
        ),
    ])


    st.divider()


    # --------------------------------------------------------
    # VISIBILITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("VR-01 | Ranking Distribution")

    visibility_df = run_filtered_query(
        select_sql="popularity_score",
        **FILTERS,
        extra_where=(
            "popularity_score IS NOT NULL"
        ),
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    sns.histplot(
        visibility_df["popularity_score"],
        bins=50,
        kde=True,
        color="forestgreen",
        ax=ax,
    )


    ax.set_title(
        "Distribution of Product Visibility (Popularity Score)"
    )

    ax.set_xlabel(
        "Popularity Score"
    )

    ax.set_ylabel("Count")


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # RATING VS POPULARITY
    # --------------------------------------------------------

    with col1:

        st.subheader("VR-02 | Rating vs Ranking")

        rating_rank = run_filtered_query(
            select_sql="""
                rating,
                popularity_score
            """,
            **FILTERS,
            extra_where=(
                "rating IS NOT NULL "
                "AND popularity_score IS NOT NULL"
            ),
        )


        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        sns.scatterplot(
            data=rating_rank,
            x="rating",
            y="popularity_score",
            alpha=0.5,
            color="darkorange",
            ax=ax,
        )


        ax.set_title(
            "Product Rating vs. Popularity Score"
        )

        ax.set_xlabel("Rating (1-5)")
        ax.set_ylabel("Popularity Score")


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # REVIEWS VS POPULARITY BUBBLE
    # --------------------------------------------------------

    with col2:

        st.subheader("VR-03 | Reviews vs Ranking")

        review_rank = run_filtered_query(
            select_sql="""
                reviews,
                popularity_score,
                price
            """,
            **FILTERS,
            extra_where=(
                "reviews IS NOT NULL "
                "AND popularity_score IS NOT NULL "
                "AND price IS NOT NULL"
            ),
        )


        if not review_rank.empty:

            sizes = review_rank["price"]


            if sizes.max() != sizes.min():

                sizes_scaled = (
                    (
                        sizes - sizes.min()
                    )
                    /
                    (
                        sizes.max()
                        - sizes.min()
                    )
                    *
                    400
                ) + 20

            else:

                sizes_scaled = pd.Series(
                    20,
                    index=review_rank.index
                )


            fig, ax = plt.subplots(
                figsize=(12, 8)
            )


            bubble_plot = ax.scatter(
                review_rank["reviews"],
                review_rank["popularity_score"],
                s=sizes_scaled,
                c=review_rank["price"],
                cmap="viridis",
                alpha=0.6,
                edgecolors="w",
            )


            ax.set_title(
                "Reviews vs. Popularity Score "
                "(Bubble Size & Color representing Price)"
            )

            ax.set_xlabel(
                "Number of Reviews"
            )

            ax.set_ylabel(
                "Popularity Score"
            )

            ax.set_xscale("log")


            cbar = fig.colorbar(
                bubble_plot,
                ax=ax
            )

            cbar.set_label(
                "Price (₹)"
            )


            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


# ============================================================
# 6. PRODUCT EXPLORER
# ============================================================

elif page == "Product Explorer":

    st.header("6. Product Explorer")


    search_text = st.text_input(
        "🔎 Search product title",
        placeholder="Type a product name or keyword...",
    )


    sort_options = {

        "Price: Low → High":
            "price ASC",

        "Price: High → Low":
            "price DESC",

        "Rating: High → Low":
            "rating DESC",

        "Reviews: High → Low":
            "reviews DESC",

        "Visibility: High → Low":
            "popularity_score DESC",

        "Title: A → Z":
            "title ASC",
    }


    sort_choice = st.selectbox(
        "Sort products",
        list(sort_options.keys()),
    )


    explorer_kpis = run_filtered_query(
        select_sql="""
            COUNT(*) AS total_products,
            AVG(price) AS avg_price,
            AVG(rating) AS avg_rating
        """,
        **FILTERS,
        search_text=search_text,
    ).iloc[0]


    metric_row([
        (
            "Total Products (Filtered)",
            number(
                explorer_kpis["total_products"]
            ),
            "Products matching all filters and search"
        ),

        (
            "Avg Price (Filtered)",
            money(
                explorer_kpis["avg_price"]
            ),
            "Average price of filtered products"
        ),

        (
            "Avg Rating (Filtered)",
            f'{explorer_kpis["avg_rating"]:.2f}'
            if pd.notna(
                explorer_kpis["avg_rating"]
            )
            else "0.00",
            "Average rating of filtered products"
        ),
    ])


    st.divider()


    # --------------------------------------------------------
    # TOP PERFORMING PRODUCTS
    # --------------------------------------------------------

    if column_exists("popularity_score"):

        top_products = run_filtered_query(
            select_sql="""
                title,
                brand,
                price,
                rating,
                reviews,
                platform,
                popularity_score
            """,
            **FILTERS,
            search_text=search_text,
            extra_where=(
                "popularity_score IS NOT NULL"
            ),
            order_by="popularity_score DESC",
            limit=5,
        )


        if not top_products.empty:

            st.subheader(
                "⭐ Top-Performing Products"
            )


            st.dataframe(
                top_products,
                use_container_width=True,
                hide_index=True,
            )


    # --------------------------------------------------------
    # PRODUCT TABLE
    # --------------------------------------------------------

    columns = [
        c
        for c in [
            "title",
            "brand",
            "price",
            "rating",
            "reviews",
            "platform",
            "discount",
        ]
        if column_exists(c)
    ]


    explorer_df = run_filtered_query(
        select_sql=", ".join(
            quote_identifier(c)
            for c in columns
        ),
        **FILTERS,
        search_text=search_text,
        order_by=sort_options[sort_choice],
        limit=500,
    )


    st.subheader(
        "Product Table"
    )


    st.caption(
        f"Showing up to 500 records. "
        f"Matching records: {int(explorer_kpis['total_products']):,}"
    )


    st.dataframe(
        explorer_df,
        use_container_width=True,
        hide_index=True,
        column_config={

            "price":
                st.column_config.NumberColumn(
                    "Price",
                    format="₹%.2f",
                ),

            "rating":
                st.column_config.NumberColumn(
                    "Rating",
                    format="%.2f",
                ),

            "reviews":
                st.column_config.NumberColumn(
                    "Reviews",
                    format="%d",
                ),

            "discount":
                st.column_config.NumberColumn(
                    "Discount",
                    format="%.2f",
                ),
        },
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Data source: SQLite table products_data. "
    "Sidebar filters are translated into parameterized SQL "
    "conditions, so KPIs, charts and the explorer update together."
)