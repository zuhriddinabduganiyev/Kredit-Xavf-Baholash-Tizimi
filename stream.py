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
warnings.filterwarnings('ignore')

# Sahifa konfiguratsiyasi
st.set_page_config(
    page_title="🏦 Kredit Xavf Baholash Tizimi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Bu kredit xavf baholash tizimi\nML algoritmlari yordamida mijozlarning kredit to'lash qobiliyatini baholaydi!"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .prediction-success {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .prediction-danger {
        background: linear-gradient(90deg, #ee0979 0%, #ff6a00 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .feature-importance {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Ma'lumotlarni yuklash
@st.cache_data
def load_data():
    local_path = "1123.xls"
    
    if os.path.exists(local_path):
        with st.spinner('📊 Ma\'lumotlar yuklanmoqda...'):
            time.sleep(1)
            try:
                data = pd.read_excel(local_path, header=1)
                st.success("✅ Ma'lumotlar muvaffaqiyatli yuklandi!")
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
        # Ustun nomlarini olish
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

# Modelni o'qitish
@st.cache_resource
def train_model(X, y):
    if X is None or y is None:
        return None, None, None, None
    
    with st.spinner('🤖 Model o\'qitilmoqda...'):
        time.sleep(2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
    
    st.success("✅ Model muvaffaqiyatli o'qitildi!")
    return model, accuracy, report, cm

# Grafiklar yaratish
def create_confusion_matrix(cm):
    fig = px.imshow(cm, 
                    labels=dict(x="Bashorat qilingan", y="Haqiqiy", color="Miqdor"),
                    x=['To\'laydi', 'To\'lamaydi'],
                    y=['To\'laydi', 'To\'lamaydi'],
                    color_continuous_scale='Blues',
                    title="🎯 Confusion Matrix")
    
    # Matnlarni qo'shish
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

def create_feature_importance_chart(model, feature_names):
    importance = model.feature_importances_
    
    # Top 10 eng muhim xususiyatlar
    indices = np.argsort(importance)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importance = [importance[i] for i in indices]
    
    fig = px.bar(x=top_importance, y=top_features, orientation='h',
                 labels={'x': 'Ahamiyat darajasi', 'y': 'Xususiyatlar'},
                 title="📊 Eng muhim xususiyatlar (Top 10)",
                 color=top_importance,
                 color_continuous_scale='viridis')
    
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

# Asosiy ilova
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Kredit Xavf Baholash Tizimi</h1>
        <p>Sun'iy intellekt yordamida mijozlarning kredit to'lash qobiliyatini baholash</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Boshqaruv Paneli")
        
        # Navigatsiya
        page = st.selectbox(
            "📋 Sahifani tanlang:",
            ["🏠 Bosh sahifa", "📊 Ma'lumotlar tahlili", "🤖 Model baholash", "🔮 Bashorat qilish", "📈 Statistika"]
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

    # Ma'lumotlarni yuklash va qayta ishlash
    data = load_data()
    if data is None:
        st.stop()
    
    X_scaled, y, scaler, feature_names = preprocess_data(data)
    if X_scaled is None:
        st.stop()
    
    model, accuracy, report, cm = train_model(X_scaled, y)
    if model is None:
        st.stop()

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
        
        # Quick insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Tezkor ma'lumotlar")
            fig_hist = px.histogram(data, x='AGE', nbins=30, 
                                  title="Yoshga nisbatan taqsimot")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Kredit limiti taqsimoti")
            fig_box = px.box(data, y='LIMIT_BAL', 
                           title="Kredit limiti (Box Plot)")
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
                st.bar_chart(missing_data[missing_data > 0])
        
        with tab2:
            selected_feature = st.selectbox("Xususiyatni tanlang:", feature_names)
            fig = px.histogram(data, x=selected_feature, color='default payment next month',
                             title=f"{selected_feature} bo'yicha taqsimot")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]  # Faqat 10ta ustun
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Korrelyatsiya matritsasi")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "🤖 Model baholash":
        st.markdown("## 🤖 Model baholash")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Confusion Matrix")
            fig_cm = create_confusion_matrix(cm)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Xususiyatlarning ahamiyati")
            fig_importance = create_feature_importance_chart(model, feature_names)
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

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🏦 Kredit Xavf Baholash Tizimi v2.0 | 
        Ishlab chiqilgan Machine Learning yordamida | 
        © 2024 Credit Risk Solutions
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()