import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- NLTK SETUP (Robust Download) ---
def download_nltk_data():
    resources = [
        'stopwords', 
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng',  
        'punkt', 
        'punkt_tab', 
        'vader_lexicon'
    ]
    
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{res}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{res}')
                except LookupError:
                    nltk.download(res, quiet=True)

download_nltk_data()

# --- CONFIGURATION ---
st.set_page_config(page_title="Hotel Intelligence Dashboard", layout="wide")

st.markdown("""
<style>
    .metric-container {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("output/booking_reviews_processed.csv")
        df['stay_date_dt'] = pd.to_datetime(df['stay_date_dt'])
        
        # Ensure stars are integers
        if 'hotel_stars' not in df.columns: df['hotel_stars'] = 0
        else: df['hotel_stars'] = df['hotel_stars'].astype(int)
        
        # Ensure category columns exist
        category_cols = ['cat_cleanliness', 'cat_location', 'cat_staff', 
                         'cat_facilities', 'cat_value', 'cat_noise']
        for col in category_cols:
            if col not in df.columns:
                df[col] = False
                
        # Ensure misclassification columns exist
        if 'neg_text_is_positive' not in df.columns:
            df['neg_text_is_positive'] = False
        if 'true_sentiment' not in df.columns:
            df['true_sentiment'] = df.get('text_sentiment_avg', 0)
            
        return df
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Data file not found. Run process_data.py first.")
        st.stop()

df = load_data()

# --- SIDEBAR Configuration ---
st.sidebar.header("🎯 Analysis Scope")
st.sidebar.write("Filter the dataset to focus on specific competitors.")

# Multi-select filter for hotels.
selected_hotel = st.sidebar.multiselect(
    "Select Hotels for Comparison", 
    options=sorted(df['hotel_name'].unique()),
    default=df['hotel_name'].unique()
)

# NEW: Category Filter
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Filter by Category")
category_map = {
    'Cleanliness': 'cat_cleanliness',
    'Location': 'cat_location', 
    'Staff': 'cat_staff',
    'Facilities': 'cat_facilities',
    'Value': 'cat_value',
    'Noise': 'cat_noise'
}
selected_categories = st.sidebar.multiselect(
    "Show reviews mentioning:",
    options=list(category_map.keys()),
    default=[]
)

# Apply filters
df_filtered = df[df['hotel_name'].isin(selected_hotel)]

# Apply category filter if selected
if selected_categories:
    category_mask = df_filtered[list(category_map.values())].any(axis=1)
    for cat in selected_categories:
        category_mask &= df_filtered[category_map[cat]]
    df_filtered = df_filtered[category_mask]

if df_filtered.empty:
    st.warning("No data selected. Please adjust your filters.")
    st.stop()
else:
    st.sidebar.success(f"✅ Analyzing {len(df_filtered)} reviews across {len(selected_hotel)} hotel(s).")

# --- MAIN DASHBOARD ---

# 1. Header & Introduction
st.title("🏨 Hotel Sentiment Engine")
st.markdown("""
This dashboard goes beyond simple star ratings. It uses **Natural Language Processing (NLP)**, specifically the **VADER** sentiment intensity analyzer, to decode *what guests actually feel* versus what score they click.
It helps identify market misalignments, hidden friction points, and the true drivers of guest satisfaction.
***
""")

# 2. KPI Section
st.subheader("📊 Executive Summary (KPIs)")

avg_rating = df_filtered['rating_score'].mean()
avg_sentiment = df_filtered['true_sentiment'].mean()  # USE TRUE SENTIMENT
total_reviews = len(df_filtered)
sentiment_gap = (avg_rating / 10) - avg_sentiment
hidden_issues = len(df_filtered[(df_filtered['rating_score'] >= 8) & (df_filtered['neg_sentiment_score'] < -0.3)])
# NEW: Misclassification count
misclassified = df_filtered['neg_text_is_positive'].sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("🅿️ Official Avg Rating", f"{avg_rating:.2f} / 10", "Booking.com Score")

col2.metric(
    "🤖 True Text Sentiment", 
    f"{avg_sentiment:.2f} / 1.0", 
    f"{sentiment_gap:.2f} Nuance Gap", 
    delta_color="inverse",
    help="""Uses combined text analysis (both positive and negative fields) for more accurate sentiment. 
The 'Nuance Gap' shows disconnect between star rating and actual text tone."""
)

col3.metric("📉 Hidden Issues", hidden_issues, "High Score + Negative Text", delta_color="inverse",
            help="Reviews with rating 8+ but negative text in description.")

col4.metric("👥 Sample Size", f"{total_reviews} Reviews")

col5.metric(
    "🔄 Misclassified", 
    f"{misclassified} ({misclassified/total_reviews*100:.1f}%)" if total_reviews > 0 else "0",
    "Positive in Neg Field",
    delta_color="off",
    help="Reviews where guests wrote POSITIVE feedback in the 'negative' section. This skews traditional sentiment analysis."
)

st.markdown("---")

# NEW: Correlation Comparison
st.subheader("📈 Sentiment Accuracy Comparison")
col_corr1, col_corr2 = st.columns([2, 1])

with col_corr1:
    corr_legacy = df_filtered[['rating_score', 'text_sentiment_avg']].corr().iloc[0,1]
    corr_true = df_filtered[['rating_score', 'true_sentiment']].corr().iloc[0,1]
    
    corr_data = pd.DataFrame({
        'Method': ['Legacy (Avg Pos/Neg)', 'True (Combined Text)'],
        'Correlation': [corr_legacy, corr_true]
    })
    
    fig_corr = px.bar(corr_data, x='Method', y='Correlation', 
                      color='Method', 
                      color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                      title="Rating vs. Sentiment Correlation",
                      text_auto='.3f')
    fig_corr.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig_corr, use_container_width=True)

with col_corr2:
    improvement = corr_true - corr_legacy
    st.metric("Correlation Improvement", f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}")
    st.info(f"""
    **Why This Matters:**
    
    The "True" sentiment method combines both text fields and ignores misleading labels. 
    
    A **{improvement:.1%} higher correlation** means our enhanced method better predicts actual guest satisfaction from text.
    """)

st.markdown("---")

# 3. MARKET STANDARD ANALYSIS
st.subheader("⭐ Star Rating vs. Guest Sentiment Nuances")

col_stars1, col_stars2 = st.columns([2, 1])

with col_stars1:
    # Aggregate data by hotel stars
    stars_group = df_filtered.groupby('hotel_stars')[['rating_score', 'true_sentiment']].mean().reset_index()
    
    fig_stars = go.Figure()
    
    # Bar Chart: Official Rating
    fig_stars.add_trace(go.Bar(
        x=stars_group['hotel_stars'], y=stars_group['rating_score'],
        name='Official Score (Click)', marker_color='#4A90E2', 
        text=round(stars_group['rating_score'], 2), textposition='auto',
        hovertemplate="Stars: %{x}<br>Official Rating: %{y:.2f}<extra></extra>"
    ))
    
    # Line Chart: AI Sentiment (Scaled to 10 for visual comparison)
    fig_stars.add_trace(go.Scatter(
        x=stars_group['hotel_stars'], y=stars_group['true_sentiment'] * 10,
        name='True Text Sentiment (Read)', line=dict(color='#FF4B4B', width=4), mode='lines+markers',
        hovertemplate="Stars: %{x}<br>Text Sentiment (Scaled): %{y:.2f}<extra></extra>"
    ))
    
    fig_stars.update_layout(
        title="Do higher stars guarantee happier guests?",
        xaxis_title="Hotel Star Standard",
        yaxis_title="Score (0-10 Scale)",
        xaxis=dict(type='category'), 
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_stars, use_container_width=True)

with col_stars2:
    st.info("""
    **💡 Key Insight: The "Perfection Detail"**
    
    This chart reveals the nuances that stars alone cannot show.
    
    * **The Gap:** Even when the **Blue Bar (Stars)** is high, the **Red Line (Sentiment)** often sits slightly lower.
    * **Strategic Takeaway:** This gap represents the "Last Mile" to perfection.
    """)

# --- CATEGORY ANALYSIS TAB (NEW) ---
st.header("📊 Detailed Insights & Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌍 Local vs Foreign", "👥 Traveler Type", "📅 Seasonality", "📁 Category Analysis", "🔍 Data Quality"])

with tab1:
    col_loc1, col_loc2 = st.columns([2, 1])
    with col_loc1:
        avg_by_origin = df_filtered.groupby('is_local')['rating_score'].mean().reset_index()
        avg_by_origin['Origin'] = avg_by_origin['is_local'].map({1: 'Domestic (Poland)', 0: 'International'})
        fig_loc = px.bar(avg_by_origin, x='Origin', y='rating_score', color='Origin', title="Rating Gap: Locals vs Tourists", text_auto='.2f', range_y=[5, 10])
        st.plotly_chart(fig_loc, use_container_width=True)
    with col_loc2:
        if len(avg_by_origin) >= 2:
            local_val = avg_by_origin.loc[avg_by_origin['is_local']==1, 'rating_score'].values
            foreign_val = avg_by_origin.loc[avg_by_origin['is_local']==0, 'rating_score'].values
            if len(local_val) > 0 and len(foreign_val) > 0:
                diff = foreign_val[0] - local_val[0]
                st.write(f"**Gap Detected:** {diff:.2f} points")
                if diff > 0.5:
                    st.warning("⚠️ **Rating Mismatch:** Locals rate significantly lower.")
                else:
                    st.success("✅ **Balanced:** Consistent experience for all guests.")

with tab2:
    avg_traveler = df_filtered.groupby('traveler_type')['rating_score'].mean().reset_index().sort_values('rating_score')
    fig_travel = px.bar(avg_traveler, x='traveler_type', y='rating_score', title="Who is the toughest critic?", color='rating_score', range_y=[6, 10], text_auto='.2f')
    st.plotly_chart(fig_travel, use_container_width=True)
    st.caption("Usually 'Couples' or 'Solo' travelers have different expectations than 'Families'.")

with tab3:
    st.markdown("#### 📅 Demand vs. Satisfaction (The 'Peak Season' Test)")
    
    df_filtered['month_num'] = df_filtered['stay_date_dt'].dt.month
    df_filtered['month_name'] = df_filtered['stay_date_dt'].dt.strftime('%b')
    
    monthly_stats = df_filtered.groupby('month_num').agg({
        'rating_score': 'mean',
        'source_id': 'count', 
        'month_name': 'first' 
    }).reset_index()
    
    month_order = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    monthly_stats['month_name'] = monthly_stats['month_num'].map(month_order)

    fig_season = make_subplots(specs=[[{"secondary_y": True}]])

    fig_season.add_trace(
        go.Bar(
            x=monthly_stats['month_name'], 
            y=monthly_stats['source_id'], 
            name="Guest Volume (Demand)",
            marker_color='rgba(50, 171, 96, 0.6)',
            hovertemplate="Reviews: %{y}<extra></extra>"
        ),
        secondary_y=False, 
    )

    fig_season.add_trace(
        go.Scatter(
            x=monthly_stats['month_name'], 
            y=monthly_stats['rating_score'], 
            name="Average Rating",
            mode='lines+markers',
            line=dict(color='darkblue', width=3),
            hovertemplate="Rating: %{y:.2f}<extra></extra>"
        ),
        secondary_y=True,
    )

    fig_season.update_layout(
        title="Does Quality Drop When It's Busy?",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    
    fig_season.update_yaxes(title_text="Number of Reviews (Demand)", secondary_y=False, showgrid=False)
    fig_season.update_yaxes(title_text="Average Rating (Quality)", secondary_y=True, range=[5, 10], showgrid=True)
    
    st.plotly_chart(fig_season, use_container_width=True)

# NEW TAB: Category Analysis
with tab4:
    st.markdown("#### 📁 Category Mention Frequency by Hotel")
    
    category_cols = ['cat_cleanliness', 'cat_location', 'cat_staff', 
                     'cat_facilities', 'cat_value', 'cat_noise']
    category_labels = ['Cleanliness', 'Location', 'Staff', 'Facilities', 'Value', 'Noise']
    
    # Overall category distribution
    col_cat1, col_cat2 = st.columns([1, 1])
    
    with col_cat1:
        cat_counts = df_filtered[category_cols].sum()
        cat_pct = (cat_counts / len(df_filtered) * 100).round(1)
        cat_df = pd.DataFrame({
            'Category': category_labels,
            'Mentions': cat_counts.values,
            'Percentage': cat_pct.values
        }).sort_values('Mentions', ascending=True)
        
        fig_cat = px.bar(cat_df, x='Mentions', y='Category', orientation='h',
                         title="What Do Guests Talk About Most?",
                         color='Mentions', color_continuous_scale='Blues',
                         text='Percentage')
        fig_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_cat2:
        # Category by sentiment
        cat_sentiment = []
        for col, label in zip(category_cols, category_labels):
            subset = df_filtered[df_filtered[col]]
            if len(subset) > 0:
                cat_sentiment.append({
                    'Category': label,
                    'Avg Sentiment': subset['true_sentiment'].mean(),
                    'Count': len(subset)
                })
        
        if cat_sentiment:
            cat_sent_df = pd.DataFrame(cat_sentiment).sort_values('Avg Sentiment')
            fig_sent = px.bar(cat_sent_df, x='Avg Sentiment', y='Category', orientation='h',
                             title="Sentiment by Category (Problem Areas = Low)",
                             color='Avg Sentiment', color_continuous_scale='RdYlGn',
                             range_color=[-0.5, 0.8])
            st.plotly_chart(fig_sent, use_container_width=True)

# NEW TAB: Data Quality / Misclassification Explorer
with tab5:
    st.markdown("#### 🔍 Misclassification Analysis")
    st.info("""
    **What is Misclassification?** 
    Sometimes guests write positive feedback in the "What didn't you like?" field (sad face icon on Booking.com).
    This confuses traditional sentiment analysis. Our enhanced pipeline detects and flags these cases.
    """)
    
    col_dq1, col_dq2 = st.columns([1, 1])
    
    with col_dq1:
        misclass_count = df_filtered['neg_text_is_positive'].sum()
        total = len(df_filtered)
        
        fig_pie = px.pie(
            values=[misclass_count, total - misclass_count],
            names=['Misclassified', 'Correct'],
            title=f"Review Classification Accuracy ({misclass_count} of {total})",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_dq2:
        # Misclassification by rating
        misclass_by_rating = df_filtered.groupby('rating_score')['neg_text_is_positive'].mean() * 100
        fig_misclass = px.bar(
            x=misclass_by_rating.index,
            y=misclass_by_rating.values,
            title="Misclassification Rate by Rating",
            labels={'x': 'Rating', 'y': 'Misclassification %'},
            color=misclass_by_rating.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_misclass, use_container_width=True)
    
    # Expandable section to view actual misclassified reviews
    with st.expander("📝 View Sample Misclassified Reviews"):
        misclassed_reviews = df_filtered[df_filtered['neg_text_is_positive']][
            ['hotel_name', 'rating_score', 'neg_text', 'neg_sentiment_score']
        ].head(15)
        
        if len(misclassed_reviews) > 0:
            st.dataframe(
                misclassed_reviews.rename(columns={
                    'hotel_name': 'Hotel',
                    'rating_score': 'Rating',
                    'neg_text': 'Text in "Negative" Field',
                    'neg_sentiment_score': 'Actual Sentiment'
                }),
                use_container_width=True
            )
        else:
            st.write("No misclassified reviews found in current filter.")

# --- SECTION D: ROOT CAUSE ANALYSIS ---
st.header("⚠️ Root Cause Analysis (Specific Complaints)")

neg_text_blob = " ".join(df_filtered['neg_text'].dropna().astype(str))
tokens = nltk.word_tokenize(neg_text_blob.lower())

# 1. POS TAGGING
tagged_tokens = nltk.pos_tag(tokens)
allowed_tags = {'NN', 'NNS', 'JJ', 'JJR', 'JJS'}

# 2. EXTENDED BAN LIST
custom_ban_list = set(stopwords.words('english')).union({
    'room', 'rooms', 'hotel', 'stay', 'everything', 'nothing', 'anything', 
    'good', 'bad', 'nice', 'great', 'ok', 'okay', 'bit', 'small', 'big',  
    'staff', 'reception', 'location', 'bathroom', 'night', 'day', 'time', 
    'wroclaw', 'poland', 'place', 'apartment', 'hostel',
    'would', 'could', 'get', 'got', 'was', 'were', 'had', 'like', 'need', 'needs', 'another', 'first' 
})

smart_tokens = [word for word, tag in tagged_tokens if tag in allowed_tags and word not in custom_ban_list and len(word) > 2]

if smart_tokens:
    most_common = Counter(smart_tokens).most_common(15)
    df_words = pd.DataFrame(most_common, columns=['Issue', 'Count'])

    fig_words = px.bar(
        df_words, x='Count', y='Issue', orientation='h',
        title="What hurts people the most",
        color='Count', color_continuous_scale='Reds'
    )
    fig_words.update_layout(yaxis=dict(autorange="reversed")) 
    st.plotly_chart(fig_words, use_container_width=True)
    st.info("""
    **💡 Data cleaning process**
    
    NLTK POS Tagging extracts only **Nouns** and **Adjectives** from complaint text,
    filtering out generic filler words to reveal specific objects of dissatisfaction.
    """)
else:
    st.warning("Not enough text data to generate meaningful topics.")

# --- FOOTER ---
st.markdown("---")
st.caption("Booking.com Sentiment Engine | Enhanced with Misclassification Detection | Powered by Python, NLTK & Streamlit")