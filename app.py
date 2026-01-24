import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- NLTK SETUP (Robust Download) ---
# This ensures all necessary linguistic models are downloaded for VADER and POS Tagging.
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
            
        return df
    except FileNotFoundError:
        st.error("CRITICAL ERROR: Data file not found. Run analysis.py first.")
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

# Apply filters
df_filtered = df[df['hotel_name'].isin(selected_hotel)]

if df_filtered.empty:
    st.warning("No data selected. Please pick a hotel.")
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
avg_sentiment = df_filtered['text_sentiment_avg'].mean()
total_reviews = len(df_filtered)
# "Sentiment Gap" calculates the disconnect between the star rating (normalized to 0-1) and the actual text sentiment.
sentiment_gap = (avg_rating / 10) - avg_sentiment
# Critical Issues: High Score (>=8) but Very Negative Text (<-0.3)
hidden_issues = len(df_filtered[(df_filtered['rating_score'] >= 8) & (df_filtered['neg_sentiment_score'] < -0.3)])

col1, col2, col3, col4 = st.columns(4)

col1.metric("🅿️ Official Avg Rating", f"{avg_rating:.2f} / 10", "Booking.com Score")

col2.metric(
    "🤖 AI Text Sentiment Score", 
    f"{avg_sentiment:.2f} / 1.0", 
    f"{sentiment_gap:.2f} Nuance Gap", 
    delta_color="inverse",
    help="""How VADER Works: This score reflects the emotional tone of the review text. 
A 'Nuance Gap' suggests that while guests are overall very happy (giving high stars), they still provide constructive feedback or mention minor issues in the text that the star rating doesn't capture."""
)

col3.metric("📉 Hidden Feedback detected", hidden_issues, "High Score + Specific Issues", delta_color="inverse",
            help="Count of reviews where guests gave a HIGH score (8+), but took the time to write about specific downsides (like smell, furniture, noise). This highlights areas for perfection in otherwise great hotels.")

col4.metric("👥 Sample Size", f"{total_reviews} Reviews Analyzed")

st.markdown("---")

# 3. MARKET STANDARD ANALYSIS
st.subheader("⭐ Star Rating vs. Guest Sentiment Nuances")

col_stars1, col_stars2 = st.columns([2, 1])

with col_stars1:
    # Aggregate data by hotel stars
    stars_group = df_filtered.groupby('hotel_stars')[['rating_score', 'text_sentiment_avg']].mean().reset_index()
    
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
        x=stars_group['hotel_stars'], y=stars_group['text_sentiment_avg'] * 10,
        name='AI Text Reality (Read)', line=dict(color='#FF4B4B', width=4), mode='lines+markers',
        hovertemplate="Stars: %{x}<br>Text Sentiment (Scaled): %{y:.2f}<extra></extra>"
    ))
    
    fig_stars.update_layout(
        title="Do higher stars guarantee happier guests?",
        xaxis_title="Hotel Star Standard",
        yaxis_title="Score (0-10 Scale)",
        # FIX: Force X-axis to be categorical to avoid messy fractional ticks (e.g., 4.5)
        xaxis=dict(type='category'), 
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_stars, use_container_width=True)

with col_stars2:
    st.info("""
    **💡 Key Insight: The "Perfection Detail"**
    
    This chart reveals the nuances that stars alone cannot show.
    
    * **The Gap:** Even when the **Blue Bar (Stars)** is high, the **Red Line (Sentiment)** often sits slightly lower. This doesn't mean the hotel is bad—it means that **even satisfied guests (who give 10/10)** often point out specific details to improve (e.g., "Great stay, but the carpet was old").
    * **Strategic Takeaway:** For hotel managers, this gap represents the "Last Mile" to perfection. For guests, it proves that reading text reviews is crucial, as a 10/10 rating often comes with small, constructive caveats regarding furniture, smell, or noise.
    """)

# --- DEMOGRAPHICS & SEASONALITY ---
st.header("📊 Detailed Insights & Hypotheses")

tab1, tab2, tab3 = st.tabs(["🌍 Local vs Foreign", "👥 Traveler Type", "📅 Seasonality"])

with tab1:
    col_loc1, col_loc2 = st.columns([2, 1])
    with col_loc1:
        avg_by_origin = df_filtered.groupby('is_local')['rating_score'].mean().reset_index()
        avg_by_origin['Origin'] = avg_by_origin['is_local'].map({1: 'Domestic (Poland)', 0: 'International'})
        fig_loc = px.bar(avg_by_origin, x='Origin', y='rating_score', color='Origin', title="Rating Gap: Locals vs Tourists", text_auto='.2f', range_y=[5, 10])
        st.plotly_chart(fig_loc, use_container_width=True)
    with col_loc2:
        diff = avg_by_origin.loc[avg_by_origin['is_local']==0, 'rating_score'].values[0] - avg_by_origin.loc[avg_by_origin['is_local']==1, 'rating_score'].values[0]
        st.write(f"**Gap Detected:** {diff:.2f} points")
        if diff > 0.5:
            st.warning("⚠️ **Rating Mismatch:** Locals rates are significantly lower.")
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
    df_filtered['month_name'] = df_filtered['stay_date_dt'].dt.strftime('%b') # Jan, Feb...
    
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
    
    st.info("""
        Here you can find few meaningful insights — for instance, you can see the busiest months of the year and when customer ratings are at their highest and lowest.
    """)

# --- SECTION D: ROOT CAUSE ANALYSIS ---
st.header("⚠️ Root Cause Analysis (Specific Complaints)")

# --- SMART FILTERING LOGIC ---
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
    
    Simple word counts are noisy. **NLTK POS Tagging** was used to perform grammatical analysis on every sentence.
    
    * Actively removed verbs, adverbs, and generic filler words (like 'also', 'get', 'hotel').
    * Extractes only **Nouns** (e.g., 'noise', 'smell', 'heating') and specific **Adjectives** (e.g., 'cold', 'dirty').
    
    This reveals the specific objects of customer dissatisfaction.
    """)
else:
    st.warning("Not enough text data to generate meaningful topics.")

# --- FOOTER ---
st.markdown("---")
st.caption("Booking.com Sentiment Engine | Powered by Python, NLTK & Streamlit")