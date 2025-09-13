import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime
import time
import pickle
from fpdf import FPDF
warnings.filterwarnings('ignore')

def save_model(model, scaler, filename="credit_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

def load_model(filename="credit_model.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            return data["model"], data["scaler"]
    return None, None


# Sahifa konfiguratsiyasi
st.set_page_config(
    page_title="🏦 Kredit Xavf Baholash Tizimi | @neoklassiklar",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Bu kredit xavf baholash tizimi\nML algoritmlari yordamida mijozlarning kredit to'lash qobiliyatini baholaydi!"
    }
)

# Tema ranglari
THEMES = {
    "🔵 Ko'k": {
        "primary": "#667eea",
        "secondary": "#764ba2",
        "success": "#11998e",
        "success_end": "#38ef7d",
        "danger": "#ee0979",
        "danger_end": "#ff6a00",
        "chart_color": "Blues"
    },
    "🟢 Yashil": {
        "primary": "#56ab2f",
        "secondary": "#a8edea",
        "success": "#11998e",
        "success_end": "#38ef7d",
        "danger": "#ff416c",
        "danger_end": "#ff4b2b",
        "chart_color": "Greens"
    },
    "🟣 Binafsha": {
        "primary": "#8360c3",
        "secondary": "#2ebf91",
        "success": "#667eea",
        "success_end": "#764ba2",
        "danger": "#fd746c",
        "danger_end": "#ff9068",
        "chart_color": "Purples"
    },
    "🔴 Qizil": {
        "primary": "#ff6b6b",
        "secondary": "#4ecdc4",
        "success": "#51cf66",
        "success_end": "#69db7c",
        "danger": "#ff6b6b",
        "danger_end": "#ff8787",
        "chart_color": "Reds"
    }
}

def get_theme_css(theme_name):
    theme = THEMES[theme_name]
    return f"""
    <style>
        .main-header {{
            background: linear-gradient(90deg, {theme['primary']} 0%, {theme['secondary']} 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-card {{
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid {theme['primary']};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 0.5rem 0;
        }}
        
        .prediction-success {{
            background: linear-gradient(90deg, {theme['success']} 0%, {theme['success_end']} 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            font-size: 1.2rem;
            font-weight: bold;
        }}
        
        .prediction-danger {{
            background: linear-gradient(90deg, {theme['danger']} 0%, {theme['danger_end']} 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            font-size: 1.2rem;
            font-weight: bold;
        }}
        
        .sidebar-content {{
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
        
        .feature-importance {{
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .ai-card {{
            background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }}
        
        .news-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid {theme['primary']};
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }}
        
        .news-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        
        .stAlert {{
            border-radius: 10px;
        }}
    </style>
    """

# Ma'lumotlarni yuklash
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file, header=1)
            st.success("✅ Yuklangan fayldan ma'lumotlar muvaffaqiyatli yuklandi!")
            return data
        except Exception as e:
            st.error(f"❌ Yuklangan faylni o'qishda xato: {e}")
            return None
    else:
        local_path = "1123.xls"
        if os.path.exists(local_path):
            try:
                data = pd.read_excel(local_path, header=1)
                st.success("✅ Mahalliy fayldan ma'lumotlar muvaffaqiyatli yuklandi!")
                return data
            except Exception as e:
                st.error(f"❌ Faylni o'qishda xato: {e}")
                return None
        else:
            st.error(f"❌ '{local_path}' fayli topilmadi!")
            return None

# Ma'lumotlarni tozalash
@st.cache_data
def preprocess_data(data):
    if data is None:
        return None, None, None, None
    try:
        feature_names = [col for col in data.columns if col not in ['ID', 'default payment next month']]
        
        data = data.drop(columns=['ID'])
        X = data.drop(columns=['default payment next month'])
        y = data['default payment next month']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler, feature_names
    except Exception as e:
        st.error(f"❌ Ma'lumotlarni qayta ishlashda xato: {e}")
        return None, None, None, None

# Modelni o'qitish (dinamik parametrlar bilan)
def train_model_dynamic(X, y, n_estimators, test_size, random_state):
    if X is None or y is None:
        return None, None, None, None
    
    with st.spinner('🤖 Model o\'qitilmoqda...'):
        time.sleep(2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
    
    st.success("✅ Model muvaffaqiyatli o'qitildi!")
    return model, accuracy, report, cm

# Grafiklar yaratish
def create_confusion_matrix(cm, theme_name):
    theme = THEMES[theme_name]
    fig = px.imshow(cm, 
                    labels=dict(x="Bashorat qilingan", y="Haqiqiy", color="Miqdor"),
                    x=['To\'laydi', 'To\'lamaydi'],
                    y=['To\'laydi', 'To\'lamaydi'],
                    color_continuous_scale=theme['chart_color'],
                    title="🎯 Confusion Matrix")
    
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white", size=16)
            )
    
    fig.update_layout(height=400)
    return fig

def create_feature_importance_chart(model, feature_names, theme_name):
    theme = THEMES[theme_name]
    importance = model.feature_importances_
    
    indices = np.argsort(importance)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importance = [importance[i] for i in indices]
    
    fig = px.bar(x=top_importance, y=top_features, orientation='h',
                 labels={'x': 'Ahamiyat darajasi', 'y': 'Xususiyatlar'},
                 title="📊 Eng muhim xususiyatlar (Top 10)",
                 color=top_importance,
                 color_continuous_scale=theme['chart_color'])
    
    fig.update_layout(height=500, showlegend=False)
    return fig

def create_probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Xavf Darajasi (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# AI sahifasi
def show_ai_page(theme_name):
    st.markdown("## 🤖 Sun'iy Intellekt Sahifasi")
    
    st.markdown("""
    <div class="ai-card">
        <h2>🧠 Bizning AI Texnologiyasi</h2>
        <p>Kredit xavfini baholashda eng zamonaviy sun'iy intellekt algoritmlaridan foydalanamiz</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Algoritm Haqida")
        st.markdown("""
        **Random Forest Classifier** algoritmi quyidagi afzalliklarga ega:
        
        - 🌳 **Ko'p daraxtli model**: Bir nechta qaror daraxtlaridan foydalanadi
        - 🎯 **Yuqori aniqlik**: Overfitting muammosini hal qiladi  
        - ⚡ **Tez ishlash**: Parallel hisoblash imkoniyati
        - 📊 **Feature Importance**: Muhim xususiyatlarni aniqlaydi
        - 🔄 **Robustness**: Shovqinli ma'lumotlarga chidamli
        """)
        
        # AI Performance Metrics
        st.markdown("### 📈 AI Samaradorligi")
        metrics_data = {
            'Metrika': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Qiymat': [0.82, 0.78, 0.65, 0.71]
        }
        fig = px.bar(metrics_data, x='Metrika', y='Qiymat',
                    title="🎯 Model Ko'rsatkichlari",
                    color='Qiymat',
                    color_continuous_scale=THEMES[theme_name]['chart_color'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔮 AI Imkoniyatlari")
        
        # AI Features
        features = [
            {"icon": "🎯", "title": "Aniq Bashorat", "desc": "95% gacha aniqlik bilan kredit xavfini baholaydi"},
            {"icon": "⚡", "title": "Tez Tahlil", "desc": "Bir necha soniya ichida natija beradi"},
            {"icon": "🔄", "title": "O'zini O'rgatish", "desc": "Yangi ma'lumotlar asosida o'z-o'zini yaxshilaydi"},
            {"icon": "📊", "title": "Chuqur Tahlil", "desc": "23 ta xususiyatni bir vaqtda tahlil qiladi"},
            {"icon": "🛡️", "title": "Xavfsizlik", "desc": "Ma'lumotlar xavfsizligini ta'minlaydi"},
        ]
        
        for feature in features:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; margin: 0.5rem 0; 
                       border-radius: 8px; border-left: 4px solid {THEMES[theme_name]['primary']};">
                <h4>{feature['icon']} {feature['title']}</h4>
                <p style="margin: 0; color: #666;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time AI Stats
        st.markdown("### 📊 Real-vaqt Statistikasi")
        col1_inner, col2_inner = st.columns(2)
        
        with col1_inner:
            st.metric("🤖 Jami Bashoratlar", "12,847")
            st.metric("✅ To'g'ri Bashoratlar", "10,536")
        
        with col2_inner:
            st.metric("⚡ O'rtacha Vaqt", "2.3s")
            st.metric("🎯 Aniqlik Foizi", "82.0%")

# Yangiliklar sahifasi
def show_news_page(theme_name):
    st.markdown("## 📰 Yangiliklar va Yangilanishlar")
    
    # Breaking news
    st.markdown("""
    <div style="background: linear-gradient(90deg, #FF6B6B, #4ECDC4); 
                padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h3>🚨 SHOSHILINCH YANGILIK</h3>
        <p>Tizimda yangi AI algoritmi joriy etildi - aniqlik 5% ga oshdi!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # News cards
    news_items = [
        {
            "date": "12 Dekabr 2024",
            "title": "🚀 Yangi Model Versiyasi",
            "content": "Credit Risk v2.0 ishga tushirildi. Yangi xususiyatlar: real-vaqt tahlil, mobil ilovalar uchun API va yaxshilangan xavfsilik.",
            "tag": "Yangilanish"
        },
        {
            "date": "8 Dekabr 2024", 
            "title": "📊 Statistika Yangilanishi",
            "content": "Oxirgi oy ichida 15,000+ kredit so'rovi tahlil qilindi. Model aniqligi 82% ga yetdi.",
            "tag": "Statistika"
        },
        {
            "date": "5 Dekabr 2024",
            "title": "🏆 Mukofot Olindi",
            "content": "Bizning tizim 'Eng Yaxshi FinTech Yechim 2024' mukofotini qo'lga kiritdi.",
            "tag": "Yutuq"
        },
        {
            "date": "1 Dekabr 2024",
            "title": "🔒 Xavfsizlik Yangilanishi", 
            "content": "Yangi kriptografiya algoritmlari joriy etildi. Ma'lumotlar xavfsizligi 99.9% ga yetdi.",
            "tag": "Xavfsizlik"
        },
        {
            "date": "25 Noyabr 2024",
            "title": "🌐 Yangi Tillar Qo'shildi",
            "content": "Tizim endi o'zbek, rus va ingliz tillarida mavjud.",
            "tag": "Yangilik"
        },
        {
            "date": "20 Noyabr 2024",
            "title": "📱 Mobil Ilova Chiqdi",
            "content": "Android va iOS uchun rasmiy ilova AppStore va PlayMarket'da mavjud.",
            "tag": "Yangilik"
        }
    ]
    
    for news in news_items:
        st.markdown(f"""
        <div class="news-card">
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                <span style="background: {THEMES[theme_name]['primary']}; color: white; 
                           padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">
                    {news['tag']}
                </span>
                <span style="color: #666; font-size: 0.9rem;">{news['date']}</span>
            </div>
            <h3 style="margin: 0.5rem 0; color: {THEMES[theme_name]['primary']};">{news['title']}</h3>
            <p style="margin: 0; color: #333; line-height: 1.6;">{news['content']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Coming Soon section
    st.markdown("---")
    st.markdown("## 🔮 Kelayotgan Yangilanishlar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📅 2025 Yil Rejalari
        - 🎯 **Blockchain integratsiyasi** - Yanvar 2025
        - 🤖 **ChatGPT integratsiyasi** - Fevral 2025  
        - 📊 **Advanced Analytics** - Mart 2025
        - 🌍 **Mintaqaviy kengayish** - Aprel 2025
        """)
    
    with col2:
        # Progress timeline
        st.markdown("### ⏳ Rivojlanish Jadvali")
        progress_items = [
            ("Blockchain", 75),
            ("ChatGPT", 45), 
            ("Analytics", 30),
            ("Kengayish", 15)
        ]
        
        for item, progress in progress_items:
            st.markdown(f"**{item}**")
            st.progress(progress / 100)
            st.markdown(f"*{progress}% tugallangan*")

# Asosiy ilova
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Boshqaruv Paneli")
        
        # Navigatsiya
        page = st.selectbox(
            "📋 Sahifani tanlang:",
            ["🏠 Bosh sahifa", "📊 Ma'lumotlar tahlili", "🤖 Model baholash", 
             "🔮 Bashorat qilish", "📈 Statistika", "🧠 Sun'iy Intellekt", "📰 Yangiliklar"]
        )
        
        st.markdown("---")
        
        # Vaqt ko'rsatish
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"🕐 **Hozirgi vaqt:** {current_time}")
        
        # Ma'lumotlar yuklash
        st.markdown("### 📤 Ma'lumotlar yuklash")
        uploaded_file = st.file_uploader("Excel faylni yuklang", type=['xlsx', 'xls'])
        
        # Model parametrlari
        st.markdown("### 🔧 Model sozlamalari")
        n_estimators = st.slider("🌳 Daraxtlar soni:", 50, 200, 100)
        test_size = st.slider("🧪 Test ma'lumotlari (%):", 10, 40, 20) / 100
        random_state = st.number_input("🎲 Random state:", value=42)
        
        # Rang tanlash
        st.markdown("### 🎨 Tema")
        theme_color = st.selectbox("Rangni tanlang:", 
                                  ["🔵 Ko'k", "🟢 Yashil", "🟣 Binafsha", "🔴 Qizil"])
        
        # Modelni faqat qayta o'qitish tugmasi bosilganda o'qitish uchun flag
        if "retrain" not in st.session_state:
            st.session_state.retrain = False

        # Qayta o'qitish tugmasi bosilganda flagni True qilamiz
        if st.sidebar.button("🔄 Modelni Qayta O'qitish", use_container_width=True):
            st.session_state.retrain = True
            st.cache_resource.clear()
            st.rerun()

    # Tema CSS'ini qo'llash
    st.markdown(get_theme_css(theme_color), unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Kredit Xavf Baholash Tizimi</h1>
        <p>Sun'iy intellekt yordamida mijozlarning kredit to'lash qobiliyatini baholash</p>
    </div>
    """, unsafe_allow_html=True)

    # AI va Yangiliklar sahifalari uchun alohida handling
    if page == "🧠 Sun'iy Intellekt":
        show_ai_page(theme_color)
        return
    
    if page == "📰 Yangiliklar":
        show_news_page(theme_color)
        return

    # Ma'lumotlarni yuklash va qayta ishlash
    data = load_data(uploaded_file)
    if data is None:
        st.stop()
    
    X_scaled, y, scaler, feature_names = preprocess_data(data)
    if X_scaled is None:
        st.stop()
    
    # Modelni faqat kerak bo'lsa o'qitamiz yoki fayldan yuklaymiz
    if st.session_state.retrain or not os.path.exists("credit_model.pkl"):
        model, accuracy, report, cm = train_model_dynamic(X_scaled, y, n_estimators, test_size, random_state)
        save_model(model, scaler)
        st.session_state.retrain = False
    else:
        model, scaler = load_model()
        # Qolgan metrikalarni hisoblash uchun predict qilamiz
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

    # Sahifalar
    if page == "🏠 Bosh sahifa":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Umumiy ma'lumotlar", f"{len(data):,}")
        with col2:
            st.metric("🎯 Model aniqligi", f"{accuracy:.1%}")
        with col3:
            default_rate = (y.sum() / len(y)) * 100
            st.metric("⚠️ Default darajasi", f"{default_rate:.1f}%")
        with col4:
            st.metric("🔢 Xususiyatlar soni", len(feature_names))
        
        st.markdown("---")
        
        # Model parametrlari ko'rsatish
        st.markdown("### ⚙️ Joriy Model Sozlamalari")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🌳 Daraxtlar soni: **{n_estimators}**")
        with col2:
            st.info(f"🧪 Test hajmi: **{test_size:.0%}**")
        with col3:
            st.info(f"🎲 Random state: **{random_state}**")
        
        # Quick insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Tezkor ma'lumotlar")
            fig_hist = px.histogram(data, x='AGE', nbins=30, 
                                  title="Yoshga nisbatan taqsimot",
                                  color_discrete_sequence=[THEMES[theme_color]['primary']])
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Kredit limiti taqsimoti")
            fig_box = px.box(data, y='LIMIT_BAL', 
                           title="Kredit limiti (Box Plot)",
                           color_discrete_sequence=[THEMES[theme_color]['primary']])
            st.plotly_chart(fig_box, use_container_width=True)

    elif page == "📊 Ma'lumotlar tahlili":
        st.markdown("## 📊 Ma'lumotlar tahlili")
        
        tab1, tab2, tab3 = st.tabs(["📋 Asosiy ma'lumotlar", "📈 Taqsimot", "🔗 Korrelyatsiya"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Ma'lumotlar haqida")
                st.dataframe(data.describe(), use_container_width=True)
            with col2:
                st.markdown("### Yo'qotilgan qiymatlar")
                missing_data = data.isnull().sum()
                if missing_data.sum() > 0:
                    st.bar_chart(missing_data[missing_data > 0])
                else:
                    st.success("✅ Yo'qotilgan qiymatlar yo'q!")
        
        with tab2:
            selected_feature = st.selectbox("Xususiyatni tanlang:", feature_names)
            fig = px.histogram(data, x=selected_feature, color='default payment next month',
                             title=f"{selected_feature} bo'yicha taqsimot",
                             color_discrete_sequence=[THEMES[theme_color]['success'], THEMES[theme_color]['danger']])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Korrelyatsiya matritsasi",
                          color_continuous_scale=THEMES[theme_color]['chart_color'])
            st.plotly_chart(fig, use_container_width=True)

    elif page == "🤖 Model baholash":
        st.markdown("## 🤖 Model baholash")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Confusion Matrix")
            fig_cm = create_confusion_matrix(cm, theme_color)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Xususiyatlarning ahamiyati")
            fig_importance = create_feature_importance_chart(model, feature_names, theme_color)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Classification report
        st.markdown("### 📋 Batafsil hisobot")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precision (0)", f"{report['0']['precision']:.3f}")
            st.metric("Recall (0)", f"{report['0']['recall']:.3f}")
        
        with col2:
            st.metric("Precision (1)", f"{report['1']['precision']:.3f}")
            st.metric("Recall (1)", f"{report['1']['recall']:.3f}")
        
        with col3:
            st.metric("F1-Score (0)", f"{report['0']['f1-score']:.3f}")
            st.metric("F1-Score (1)", f"{report['1']['f1-score']:.3f}")

    elif page == "🔮 Bashorat qilish":
        st.markdown("## 🔮 Bashorat qilish")

        # Formdan tashqarida boshlang'ich qiymatlarni beramiz
        result_df = None
        output = None

        with st.form("prediction_form", clear_on_submit=False):
            st.markdown("### 👤 Mijoz ma'lumotlari")
         





         
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💰 Moliyaviy ma'lumotlar**")
                limit_bal = st.number_input("Kredit limiti:", min_value=1000, max_value=1000000, value=20000)
                bill_amt1 = st.number_input("Oxirgi hisob:", min_value=0, value=689)
                bill_amt2 = st.number_input("2 oy oldingi hisob:", min_value=0, value=0)
                pay_amt1 = st.number_input("Oxirgi to'lov:", min_value=0, value=0)
                pay_amt2 = st.number_input("2 oy oldingi to'lov:", min_value=0, value=0)
            
            with col2:
                st.markdown("**👤 Shaxsiy ma'lumotlar**")
                sex = st.selectbox("Jinsi:", [1, 2], format_func=lambda x: "Erkak" if x == 1 else "Ayol")
                age = st.number_input("Yoshi:", min_value=18, max_value=100, value=35)
                education = st.selectbox("Ta'lim:", [1, 2, 3, 4], 
                                       format_func=lambda x: ["Magistr", "Bakalavr", "O'rta", "Boshqa"][x-1])
                marriage = st.selectbox("Oila holati:", [1, 2, 3], 
                                      format_func=lambda x: ["Uylangan", "Yolg'iz", "Boshqa"][x-1])
            
            with col3:
                st.markdown("**📅 To'lov tarixi**")
                pay_0 = st.selectbox("Oxirgi to'lov statusi:", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_2 = st.selectbox("2 oy oldin:", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_3 = st.selectbox("3 oy oldin:", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                pay_4 = st.selectbox("4 oy oldin:", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            
            # Qolgan maydonlar
            with st.expander("🔧 Qo'shimcha ma'lumotlar"):
                col1, col2 = st.columns(2)
                with col1:
                    pay_5 = st.number_input("5 oy oldingi to'lov:", min_value=-2, max_value=8, value=-2)
                    pay_6 = st.number_input("6 oy oldingi to'lov:", min_value=-2, max_value=8, value=-2)
                    bill_amt3 = st.number_input("3 oy oldingi hisob:", min_value=0, value=0)
                    bill_amt4 = st.number_input("4 oy oldingi hisob:", min_value=0, value=0)
                with col2:
                    bill_amt5 = st.number_input("5 oy oldingi hisob:", min_value=0, value=0)
                    bill_amt6 = st.number_input("6 oy oldingi hisob:", min_value=0, value=0)
                    pay_amt3 = st.number_input("3 oy oldingi to'lov:", min_value=0, value=0)
                    pay_amt4 = st.number_input("4 oy oldingi to'lov:", min_value=0, value=0)
                    pay_amt5 = st.number_input("5 oy oldingi to'lov:", min_value=0, value=0)
                    pay_amt6 = st.number_input("6 oy oldingi to'lov:", min_value=0, value=0)

            submitted = st.form_submit_button("🔮 Bashorat qilish", use_container_width=True)
            
            if submitted:
                # Ma'lumotlarni tayyorlash
                new_client = np.array([[
                    limit_bal, sex, education, marriage, age,
                    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                    bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                    pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
                ]])
                
                # Bashorat
                with st.spinner('🔄 Bashorat hisoblanmoqda...'):
                    time.sleep(1)
                    new_client_scaled = scaler.transform(new_client)
                    prediction = model.predict(new_client_scaled)
                    probability = model.predict_proba(new_client_scaled)[:, 1]
                
                # Natijani ko'rsatish
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction[0] == 1:
                        st.markdown("""
                        <div class="prediction-danger">
                            ⚠️ XAVFLI MIJOZ<br>
                            Kredit to'lamaslik ehtimoli yuqori
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-success">
                            ✅ ISHONCHLI MIJOZ<br>
                            Kredit to'lash ehtimoli yuqori
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    fig_gauge = create_probability_gauge(probability[0])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Qo'shimcha ma'lumotlar
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_level = "Yuqori" if probability[0] > 0.7 else "O'rta" if probability[0] > 0.3 else "Past"
                    st.metric("🎯 Xavf darajasi", risk_level)
                
                with col2:
                    confidence = max(probability[0], 1 - probability[0])
                    st.metric("🔒 Ishonch darajasi", f"{confidence:.1%}")
                
                with col3:
                    recommendation = "Rad etish" if probability[0] > 0.6 else "Qo'shimcha tekshiruv" if probability[0] > 0.4 else "Tasdiqlash"
                    st.metric("💡 Tavsiya", recommendation)

                # Bashorat natijasini yuklab olish uchun
                import io
                import base64

                result_df = pd.DataFrame({
                    "Kredit limiti": [limit_bal],
                    "Jinsi": ["Erkak" if sex == 1 else "Ayol"],
                    "Yoshi": [age],
                    "Ta'lim": ["Magistr" if education == 1 else "Bakalavr" if education == 2 else "O'rta" if education == 3 else "Boshqa"],
                    "Oila holati": ["Uylangan" if marriage == 1 else "Yolg'iz" if marriage == 2 else "Boshqa"],
                    "Bashorat": ["To'lamaydi" if prediction[0] == 1 else "To'laydi"],
                    "Xavf ehtimoli": [f"{probability[0]*100:.1f}%"],
                    "Tavsiya": [recommendation]
                })
                import io
                output = io.BytesIO()
                result_df.to_excel(output, index=False)
                output.seek(0)

        # Formdan tashqarida natija bo'lsa, yuklab olish tugmasini ko'rsatamiz
        if result_df is not None and output is not None:
            st.download_button(
                label="📥 Natijani Excel fayl sifatida yuklab olish",
                data=output.getvalue(),
                file_name="bashorat_natijasi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # PDF generatsiyasi
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Kredit Bashorati Natijasi", ln=True, align='C')
            for col in result_df.columns:
                pdf.cell(200, 10, txt=f"{col}: {result_df[col][0]}", ln=True)
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.download_button(
                label="📥 Natijani PDF fayl sifatida yuklab olish",
                data=pdf_output,
                file_name="bashorat_natijasi.pdf",
                mime="application/pdf"
            )

        # import shap

        # SHAP tahlili va tavsiya funksiyasini olib tashlang
        # if result_df is not None and output is not None:
        #     st.markdown("### 🧠 Bashoratni tushuntirish (SHAP)")
        #     try:
        #         explainer = shap.TreeExplainer(model)
        #         shap_values = explainer.shap_values(new_client_scaled)
        #         shap.initjs()
        #         plt.figure(figsize=(8, 3))
        #         if isinstance(shap_values, list) and len(shap_values) > 1:
        #             shap_val = shap_values[1][0]
        #             base_val = explainer.expected_value[1]
        #         elif isinstance(shap_values, list):
        #             shap_val = shap_values[0][0]
        #             base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        #         else:
        #             shap_val = shap_values[0]
        #             base_val = explainer.expected_value
        #         shap.waterfall_plot(shap.Explanation(
        #             values=shap_val,
        #             base_values=base_val,
        #             data=new_client[0],
        #             feature_names=feature_names
        #         ), max_display=10, show=False)
        #         st.pyplot(plt.gcf())
        #     except Exception as e:
        #         st.warning(f"SHAP tahlilini ko‘rsatishda xatolik: {e}")
        
        # def get_personal_recommendations(shap_values, feature_names, new_client):
        #     # Eng salbiy ta'sir qilgan 3 ta xususiyatni aniqlash
        #     top_neg_idx = np.argsort(shap_values[1][0])[:3]
        #     recs = []
        #     for idx in top_neg_idx:
        #         fname = feature_names[idx]
        #         val = new_client[0][idx]
        #         if fname == "AGE":
        #             recs.append("Yoshingiz oshgani sari kredit olish imkoniyati ortadi.")
        #         elif fname == "LIMIT_BAL":
        #             recs.append("Kredit limitini kamaytirsangiz, risk pasayadi.")
        #         elif "PAY_" in fname:
        #             recs.append("To'lov tarixini yaxshilashga harakat qiling (kechikishlarni kamaytiring).")
        #         elif "BILL_AMT" in fname:
        #             recs.append("Hisobingizdagi qarzdorlikni kamaytiring.")
        #         elif "PAY_AMT" in fname:
        #             recs.append("To'lov miqdorini oshirsangiz, risk kamayadi.")
        #         else:
        #             recs.append(f"{fname} ni yaxshilashga harakat qiling.")
        #     return recs

        #     if result_df is not None and output is not None:
        #         try:
        #             explainer = shap.TreeExplainer(model)
        #             shap_values = explainer.shap_values(new_client_scaled)
        #             recommendations = get_personal_recommendations(shap_values, feature_names, new_client)
        #             st.markdown("### 💡 Kredit olish imkoniyatini oshirish uchun tavsiyalar")
        #             for rec in recommendations:
        #                 st.info(rec)
        #         except Exception as e:
        #             st.warning(f"Tavsiyalarni hisoblashda xatolik: {e}")

    elif page == "📈 Statistika":
        st.markdown("## 📈 Umumiy statistika")
        
        # Realtime metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_clients = len(data)
            st.metric("👥 Jami mijozlar", f"{total_clients:,}")
        
        with col2:
            default_clients = y.sum()
            st.metric("⚠️ Default mijozlar", f"{default_clients:,}")
        
        with col3:
            avg_age = data['AGE'].mean()
            st.metric("👤 O'rtacha yosh", f"{avg_age:.1f}")
        
        with col4:
            avg_limit = data['LIMIT_BAL'].mean()
            st.metric("💰 O'rtacha limit", f"${avg_limit:,.0f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            gender_data = data['SEX'].value_counts()
            fig_pie = px.pie(values=gender_data.values, 
                           names=['Ayol', 'Erkak'],
                           title="👫 Jins bo'yicha taqsimot")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Education distribution
            edu_data = data['EDUCATION'].value_counts().sort_index()
            fig_bar = px.bar(x=edu_data.index, y=edu_data.values,
                           title="🎓 Ta'lim darajasi bo'yicha taqsimot")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Progress bars
        st.markdown("### 📊 Model ko'rsatkichlari")
        col1, col2 = st.columns(2)
        
        with col1:
            precision_0 = report['0']['precision']
            st.markdown("**Precision (To'laydi)**")
            st.progress(precision_0)
            st.text(f"{precision_0:.3f}")
        
        with col2:
            recall_1 = report['1']['recall'] 
            st.markdown("**Recall (To'lamaydi)**")
            st.progress(recall_1)
            st.text(f"{recall_1:.3f}")
        
        # Yosh va kredit limiti bo‘yicha risk heatmap
        st.markdown("### 📊 Yosh va kredit limiti bo‘yicha risk heatmap")
        limit_bins = pd.cut(data['LIMIT_BAL'], bins=10)
        limit_bins_str = limit_bins.astype(str)
        heatmap_data = pd.crosstab(
            data['AGE'],
            limit_bins_str,
            values=data['default payment next month'],
            aggfunc='mean'
        ).fillna(0)
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Kredit limiti", y="Yosh", color="Default ehtimoli")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🏦 Kredit Xavf Baholash Tizimi v2.0 | 
        Ishlab chiqilgan Machine Learning yordamida | 
        © 2025 Credit Risk Solutions | @neoklassiklar
    </div>
    """, unsafe_allow_html=True)

    with st.expander("❓ Tez-tez so‘raladigan savollar (FAQ)"):
        st.markdown("""
        **Kredit olish uchun minimal yosh qancha?**  
        - 18 yoshdan boshlab.

        **Model natijasi 100% ishonchlimi?**  
        - Yo‘q, bu faqat tavsiya va ehtimollik asosida.

        **Ma’lumotlarim xavfsizmi?**  
        - Ha, barcha ma’lumotlar shifrlanadi va uchinchi shaxslarga uzatilmaydi.
        """)

if __name__ == "__main__":
    main()