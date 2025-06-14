import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Initial Setup and Configuration for Elden Ring Theme ---
# The page_icon is now the URL you provided
st.set_page_config(layout="wide", page_title="Analisis Sentimen Elden Ring", page_icon="asset/icon.jpg")
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

@st.cache_data
def load_data():
    """Loads the game review dataset."""
    df = pd.read_csv("pre_df_export.csv")
    df['review'] = df['review'].astype(str)
    df['ready_review'] = df['ready_review'].astype(str)
    return df

@st.cache_resource
def load_models():
    """Loads the pre-trained sentiment analysis models."""
    loaded_svm_pipeline = joblib.load('SMOTE_best_svm_pipeline.joblib')
    loaded_rf_pipeline = joblib.load('SMOTE_best_rf_pipeline.joblib')
    analyzer = SentimentIntensityAnalyzer()
    return loaded_svm_pipeline, loaded_rf_pipeline, analyzer

# Load all resources
df_reviews = load_data()
model_svm, model_rf, vader_analyzer = load_models()

# --- UI Styling ---
st.markdown("""
<style>
/* This CSS targets the radio button to make it look more like tabs */
div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}
div.row-widget.stRadio > div > label > div {
    background-color: #F0F2F6;
    padding: 10px 24px;
    border-radius: 4px 4px 0px 0px;
    margin: 0px !important;
}
div.row-widget.stRadio > div > label > div:hover {
    background-color: #E0E2E6;
}
div.row-widget.stRadio > div > label[data-baseweb="radio"] > div:first-child {
    background-color: #FFFFFF !important;
    border-bottom: 2px solid #FFFFFF;
}
.positive-word {
    background-color: #C8E6C9; /* light green */
    color: #256029;
    padding: 2px 6px;
    border-radius: 4px;
}

.negative-word {
    background-color: #FFCDD2; /* light red */
    color: #B71C1C;
    padding: 2px 6px;
    border-radius: 4px;
}
</style>""", unsafe_allow_html=True)

# --- Helper Functions (No Changes Here) ---
def predict_sentiments(filtered_df):
    filtered_df['svm_sentiment'] = model_svm.predict(filtered_df['ready_review'])
    filtered_df['rf_sentiment'] = model_rf.predict(filtered_df['ready_review'])
    filtered_df['vader_compound'] = filtered_df['ready_review'].apply(lambda text: vader_analyzer.polarity_scores(text)['compound'])
    def classify_vader(score):
        if score >= 0.05: return 'positive'
        if score <= -0.05: return 'negative'
        return 'neutral'
    filtered_df['vader_sentiment'] = filtered_df['vader_compound'].apply(classify_vader)
    return filtered_df

def highlight_sentiment_words(text):
    highlighted_text = []
    words = str(text).split()
    for word in words:
        score = vader_analyzer.polarity_scores(word)['compound']
        if score > 0.1: highlighted_text.append(f"<span class='positive-word'>{word}</span>")
        elif score < -0.1: highlighted_text.append(f"<span class='negative-word'>{word}</span>")
        else: highlighted_text.append(word)
    return " ".join(highlighted_text)

def generate_wordcloud(text_series, title):
    if text_series.empty:
        st.write(f"Tidak ada data untuk membuat word cloud {title}.")
        return
    text = " ".join(review for review in text_series)
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# --- Main Application UI ---
with st.sidebar:
    st.image("asset/Elden-Ring-Logo.png", use_container_width=True)
    st.title("Panel Ulasan Elden Ring")
    keyword = st.text_input("Cari Topik dalam Ulasan", placeholder="Masukan Kata Kunci yang berkaitan dengan Elden Ring", help="Cari ulasan tentang bos, area, atau fitur tertentu.")
    analyze_button = st.button("Analisa Sentimen", type="primary", use_container_width=True)

st.title("Analisis Sentimen Elden Ring")
st.markdown("Arise, ye Tarnished! Selamat datang di dashboard untuk menganalisis sentimen para pemain di The Lands Between. Masukkan **kata kunci di sidebar** untuk mengungkap apa yang dikatakan para pemain lain tentang petualangan mereka.")

if not analyze_button and 'results_df' not in st.session_state:
     st.info("Menunggu input kata kunci dari seorang Tarnished di sidebar untuk memulai analisis...")
     st.stop()
     
if analyze_button and not keyword.strip():
    st.error("Harap masukkan kata kunci, wahai Tarnished.")
    st.stop()

if analyze_button:
    with st.spinner(f"Memanggil para peramal untuk menganalisis ulasan: '{keyword}'..."):
        filtered_df = df_reviews[df_reviews['ready_review'].str.contains(keyword.lower(), na=False)].copy()
        if filtered_df.empty:
            st.warning(f"Tidak ada ulasan yang ditemukan mengandung kata kunci '{keyword}'. Coba kata kunci lain.")
            st.stop()
        results_df = predict_sentiments(filtered_df)
        st.session_state['results_df'] = results_df
        st.session_state['keyword'] = keyword

if 'results_df' in st.session_state:
    results_df = st.session_state['results_df']
    keyword = st.session_state['keyword']
    
    st.success(f"Analisis selesai! Menampilkan hasil untuk **{len(results_df)} ulasan** terkait **'{keyword}'**.")
    color_map = {'positive': '#2ca02c', 'neutral': '#ff7f0e', 'negative': '#d62728'}
    
    # --- LOGIC CHANGE: Replacing st.tabs with st.radio for state persistence ---
    tab_options = ["Ringkasan Utama", "Perbandingan Model", "Eksplorasi Ulasan"]
    selected_tab = st.radio("Navigasi Halaman:", tab_options, horizontal=True, label_visibility="collapsed")

    # --- TAB 1 CONTENT ---
    if selected_tab == "Ringkasan Utama":
        st.header(f"Gambaran Umum Sentimen untuk '{keyword}'")
        st.caption("Ringkasan ini didasarkan pada analisis VADER, sebuah model yang menganalisis sentimen berdasarkan aturan pada kata dan emoji.")
        vader_counts = results_df['vader_sentiment'].value_counts()
        total_reviews = len(results_df)
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment not in vader_counts: vader_counts[sentiment] = 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Total Ulasan", value=total_reviews)
        col2.metric(label="👍 Ulasan Positif", value=vader_counts.get('positive', 0))
        col3.metric(label="😐 Ulasan Netral", value=vader_counts.get('neutral', 0))
        col4.metric(label="👎 Ulasan Negatif", value=vader_counts.get('negative', 0))
        
        st.markdown("---")
        
        col_chart, col_words = st.columns([1, 2])
        with col_chart:
            st.subheader("Distribusi Sentimen")
            fig_donut = go.Figure(data=[go.Pie(labels=vader_counts.index, values=vader_counts.values, hole=.4, marker_colors=[color_map.get(x) for x in vader_counts.index])])
            fig_donut.update_layout(showlegend=True, height=400, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_donut, use_container_width=True)
        with col_words:
            st.subheader("Topik yang Sering Dibicarakan")
            positive_text = results_df[results_df['vader_sentiment'] == 'positive']['ready_review']
            st.write("**Kata Kunci dalam Ulasan Positif**")
            generate_wordcloud(positive_text, "Positif")
            negative_text = results_df[results_df['vader_sentiment'] == 'negative']['ready_review']
            st.write("**Kata Kunci dalam Ulasan Negatif**")
            generate_wordcloud(negative_text, "Negatif")

    # --- TAB 2 CONTENT ---
    elif selected_tab == "Perbandingan Model":
        st.header("Perbandingan Detail Antar Model Analisis")
        st.markdown("Bandingkan hasil dari setiap model untuk melihat perbedaannya dalam mengklasifikasikan sentimen. Ini lebih teknis dan menunjukkan bagaimana model AI bisa memiliki 'opini' yang berbeda.")
        st.subheader("Distribusi Sentimen per Model")
        col1, col2, col3 = st.columns(3)
        models = {'VADER': 'vader_sentiment', 'SVM (AI Model)': 'svm_sentiment', 'Random Forest (AI Model)': 'rf_sentiment'}
        for i, (model_name, col_name) in enumerate(models.items()):
            container = [col1, col2, col3][i]
            with container:
                counts = results_df[col_name].value_counts()
                fig_donut = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=.4, marker_colors=[color_map.get(x) for x in counts.index])])
                fig_donut.update_layout(title_text=f'<b>{model_name}</b>', annotations=[dict(text=f'{len(results_df)}', x=0.5, y=0.5, font_size=20, showarrow=False)], showlegend=False, height=300, margin=dict(t=50, b=0, l=0, r=0))
                st.plotly_chart(fig_donut, use_container_width=True)
        st.subheader("Perbandingan Jumlah Ulasan per Sentimen")
        summary_df = pd.DataFrame({'VADER': results_df['vader_sentiment'].value_counts(), 'SVM': results_df['svm_sentiment'].value_counts(), 'Random Forest': results_df['rf_sentiment'].value_counts()}).reindex(['positive', 'neutral', 'negative']).fillna(0).astype(int)
        fig_bar = px.bar(summary_df.T, barmode='group', color_discrete_map=color_map, labels={'value': 'Jumlah Ulasan', 'index': 'Model', 'variable': 'Sentimen'})
        fig_bar.update_layout(title_text="Perbandingan Agregat Model")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 3 CONTENT ---
    elif selected_tab == "Eksplorasi Ulasan":
        st.header("Eksplorasi Ulasan dengan *Highlight* Kata")
        st.markdown("Lihat ulasan individual dan perhatikan kata-kata kunci yang memengaruhi skor sentimennya (berdasarkan VADER).")
        sentiment_filter = st.selectbox("Filter berdasarkan sentimen (VADER):", options=['Semua', 'positive', 'neutral', 'negative'])
        
        if sentiment_filter == 'Semua':
            display_df = results_df
        else:
            display_df = results_df[results_df['vader_sentiment'] == sentiment_filter]

        if display_df.empty:
            st.info(f"Tidak ada ulasan dengan sentimen '{sentiment_filter}' untuk kata kunci ini.")
        else:
            total_reviews = len(df_reviews[df_reviews['ready_review'].str.contains(keyword.lower(), na=False)])
            st.write(f"Menampilkan {len(display_df)} dari {total_reviews} ulasan:")
            for _, row in display_df.head(50).iterrows():
                with st.expander(f"**Ulasan (VADER: {row['vader_sentiment'].capitalize()})** - *Voted Up: {row['votes_up']}*"):
                    st.markdown("##### Teks Ulasan dengan *Highlight* Sentimen:")
                    highlighted_review = highlight_sentiment_words(row['review'])
                    st.markdown(highlighted_review, unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("##### Detail Prediksi Model Lain:")
                    st.json({"VADER (Aturan Kata)": f"{row['vader_sentiment']} (Skor: {row['vader_compound']:.2f})", "SVM (AI Model)": row['svm_sentiment'], "Random Forest (AI Model)": row['rf_sentiment']})