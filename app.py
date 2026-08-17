import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AGRINEX AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL LIGHT-MODE STYLING
# ============================================================
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .block-container { padding-top: 2rem; }

    .agx-hero {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8F2 100%);
        border: 1px solid #C8E6C9;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .agx-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 14px;
    }
    .agx-result {
        background: #E8F5E9;
        border: 1px solid #A5D6A7;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .agx-badge {
        display:inline-block; font-size:11px; font-weight:700;
        padding:3px 10px; border-radius:999px; letter-spacing:0.03em;
    }
    .badge-live { background:#C8E6C9; color:#1B5E20; }
    .badge-demo { background:#FFE0B2; color:#8D5A00; }

    .agx-metric { text-align:center; padding:14px; border-right:1px solid #EEE; }
    .agx-metric:last-child { border-right:none; }
    .agx-metric .num { font-size:26px; font-weight:800; color:#2E7D32; }
    .agx-metric .lbl { font-size:12px; color:#777; margin-top:2px; }

    section[data-testid="stSidebar"] { background-color: #F7FBF7; border-right: 1px solid #E0E0E0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD REAL MODEL (crop recommendation) — falls back to a
# rule-based demo if the .pkl files aren't in this folder yet
# ============================================================
MODEL_PATH = "crop_model.pkl"
LABELS_PATH = "crop_labels.json"
model = None
crop_labels = []
model_is_real = False

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        with open(LABELS_PATH) as f:
            crop_labels = json.load(f)
        model_is_real = True
    except Exception as e:
        st.sidebar.warning(f"Model file found but couldn't load: {e}")

# ============================================================
# GROQ CLIENT (shared) — used by Tab 1 crop-agent explanation
# and Tab 7 farmer assistant chat
# ============================================================
from groq import Groq

groq_client = None
groq_ready = False
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=groq_key)
    groq_ready = True
except Exception:
    groq_ready = False

# ============================================================
# CROP ADVISOR AGENT (ML + RAG + LLM) — cached so the TF-IDF
# index over crop_calendar_kb.json isn't rebuilt on every rerun
# ============================================================
from crop_agent import CropAdvisorAgent

@st.cache_resource
def get_crop_agent(_client):
    return CropAdvisorAgent(_client, kb_path="crop_calendar_kb.json")

crop_agent = get_crop_agent(groq_client) if groq_ready else None

CROP_ICONS = {
    "rice":"🌾","maize":"🌽","chickpea":"🫘","kidneybeans":"🫘","pigeonpeas":"🫛",
    "mothbeans":"🫘","mungbean":"🫘","blackgram":"🫘","lentil":"🫛","pomegranate":"🍎",
    "banana":"🍌","mango":"🥭","grapes":"🍇","watermelon":"🍉","muskmelon":"🍈",
    "apple":"🍎","orange":"🍊","papaya":"🌴","coconut":"🥥","cotton":"☁️",
    "jute":"🧵","coffee":"☕"
}

def rule_based_predict(n,p,k,temp,hum,ph,rain):
    ranges = {
      'rice':dict(n=(60,100),p=(35,60),k=(35,45),temp=(20,27),hum=(75,90),ph=(5.5,7),rain=(180,300)),
      'maize':dict(n=(60,100),p=(35,60),k=(15,25),temp=(18,27),hum=(55,75),ph=(5.5,7.5),rain=(60,110)),
      'chickpea':dict(n=(20,60),p=(55,80),k=(75,100),temp=(17,25),hum=(14,20),ph=(6,8),rain=(60,105)),
      'cotton':dict(n=(100,140),p=(35,60),k=(15,25),temp=(22,26),hum=(75,85),ph=(5.5,8),rain=(60,110)),
      'coconut':dict(n=(0,35),p=(0,35),k=(25,35),temp=(25,30),hum=(90,100),ph=(5,7),rain=(130,230)),
      'banana':dict(n=(80,120),p=(70,100),k=(45,55),temp=(25,30),hum=(75,85),ph=(5.5,6.5),rain=(90,150)),
      'mango':dict(n=(0,35),p=(15,35),k=(25,35),temp=(27,32),hum=(45,55),ph=(5.5,7),rain=(35,100)),
      'watermelon':dict(n=(80,120),p=(0,15),k=(45,55),temp=(24,27),hum=(80,90),ph=(6,7),rain=(35,55)),
      'lentil':dict(n=(0,30),p=(60,80),k=(15,25),temp=(18,30),hum=(60,70),ph=(6,7),rain=(35,55)),
    }
    vals = dict(n=n,p=p,k=k,temp=temp,hum=hum,ph=ph,rain=rain)
    def sc(v, r):
        lo,hi = r
        if lo<=v<=hi: return 1.0
        span = hi-lo or 1
        return max(0, 1-(lo-v if v<lo else v-hi)/span)
    scored = []
    for crop, r in ranges.items():
        s = np.mean([sc(vals[k_], r[k_]) for k_ in vals])
        scored.append((crop, s))
    scored.sort(key=lambda x:-x[1])
    return scored

# ============================================================
# SIDEBAR — Farmer profile (feeds the whole app)
# ============================================================
with st.sidebar:
    st.markdown("### 🌾 AGRINEX AI")
    st.caption("Farmer Digital Profile")
    farmer_name = st.text_input("Farmer name", "Ramesh Patil")
    farmer_id = st.text_input("Farmer ID", "AGX-10234")
    location = st.text_input("Farm location", "Nashik, Maharashtra")
    land_area = st.number_input("Land area (acres)", min_value=0.1, value=2.5, step=0.1)
    st.divider()
    st.caption(f"Model status: {'🟢 Live model loaded' if model_is_real else '🟡 Demo mode (add crop_model.pkl + crop_labels.json to this folder for the real model)'}")
    st.divider()
    st.caption("AGRINEX AI — prototype build")

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="agx-hero">
    <h1 style="margin:0; color:#1B5E20;">🌾 AGRINEX AI</h1>
    <p style="margin:6px 0 0 0; color:#33512E; font-size:16px;">
    Namaskar, <b>{farmer_name}</b> — {location} | {land_area} acres | ID: {farmer_id}
    </p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
for col, num, lbl in [
    (c1,"9","AI systems"), (c2,"22","Crops supported"),
    (c3,"7","Live soil/climate inputs"), (c4,"3","Languages supported")
]:
    with col:
        st.markdown(f'<div class="agx-metric"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.write("")

# ============================================================
# TABS — one per planned system
# ============================================================
tabs = st.tabs([
    "🌱 Crop Recommendation", "🍃 Disease Detection", "📈 Yield Prediction",
    "💰 Price Forecast", "☁️ Weather", "🧪 Soil Analysis",
    "💬 Farmer Assistant", "🛒 Marketplace"
])

# ---------------- TAB 1: CROP RECOMMENDATION (REAL MODEL) ----------------
with tabs[0]:
    badge = '<span class="agx-badge badge-live">LIVE MODEL</span>' if model_is_real else '<span class="agx-badge badge-demo">DEMO MODE</span>'
    st.markdown(f"#### Crop Recommendation {badge}", unsafe_allow_html=True)
    st.caption("Random Forest trained on soil nutrients + climate → best-fit crop.")

    colL, colR = st.columns([1,1])
    with colL:
        st.markdown('<div class="agx-card">', unsafe_allow_html=True)
        n = st.slider("Nitrogen (N)", 0, 140, 90)
        p = st.slider("Phosphorus (P)", 0, 145, 42)
        k = st.slider("Potassium (K)", 0, 205, 43)
        temp = st.slider("Temperature (°C)", 0.0, 45.0, 20.8)
        hum = st.slider("Humidity (%)", 0.0, 100.0, 82.0)
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 202.9)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid/Summer"])
        run = st.button("🔍 Recommend Crop", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with colR:
        if run:
            if model_is_real:
                sample = pd.DataFrame([{'N':n,'P':p,'K':k,'temperature':temp,'humidity':hum,'ph':ph,'rainfall':rain}])
                pred = model.predict(sample)[0]
                proba = model.predict_proba(sample)[0]
                conf = max(proba)*100
                top_idx = np.argsort(proba)[-3:][::-1]
                top3 = [(model.classes_[i], proba[i]*100) for i in top_idx]
            else:
                scored = rule_based_predict(n,p,k,temp,hum,ph,rain)
                pred, conf = scored[0][0], scored[0][1]*100
                top3 = [(c, s*100) for c,s in scored[:3]]

            icon = CROP_ICONS.get(pred, "🌱")
            st.markdown(f"""
            <div class="agx-result">
                <div style="font-size:40px;">{icon}</div>
                <div style="font-size:24px; font-weight:800; color:#1B5E20; text-transform:capitalize;">{pred}</div>
                <div style="font-size:13px; color:#558B2F;">Confidence: {conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.markdown("**Top 3 matches**")
            for c, s in top3:
                st.write(f"{CROP_ICONS.get(c,'🌱')} **{c}** — {s:.1f}%")
                st.progress(min(1.0, s/100))

            st.write("")
            st.markdown("**🤖 AI Agent's take** <span class=\"agx-badge badge-live\">AGENT + RAG</span>", unsafe_allow_html=True)
            if crop_agent is not None:
                with st.spinner("Agent reasoning over ML result + regional crop calendar..."):
                    try:
                        result = crop_agent.recommend(
                            ml_top3=top3,
                            inputs={"N": n, "P": p, "K": k, "temperature": temp,
                                    "humidity": hum, "ph": ph, "rainfall": rain},
                            location=location,
                            season=season,
                        )
                        st.info(result["explanation"])
                        with st.expander("📚 Regional knowledge the agent used"):
                            if result["retrieved_context"]:
                                for r in result["retrieved_context"]:
                                    st.caption(f"**{r['state']} — {r['season']}** (match score {r['score']:.2f})")
                                    st.write(r["text"])
                            else:
                                st.caption("No specific regional match found for this location/season.")
                    except Exception as e:
                        st.warning(f"Agent explanation unavailable right now: {e}")
            else:
                st.caption("Add GROQ_API_KEY to .streamlit/secrets.toml (or Streamlit Cloud settings) to enable the AI agent's explanation here.")
        else:
            st.info("Sliders adjust kar and click **Recommend Crop**.")

# ---------------- TAB 2: DISEASE DETECTION (DEMO) ----------------
with tabs[1]:
    st.markdown('#### Crop Disease Detection <span class="agx-badge badge-demo">DEMO</span>', unsafe_allow_html=True)
    st.caption("Upload a leaf photo. In production this calls a CNN/Vision Transformer trained on plant-disease imagery.")
    crop_choice = st.selectbox("Crop", ["Tomato","Potato","Cotton","Wheat","Rice","Grapes"])
    img = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])
    if img:
        c1,c2 = st.columns([1,1])
        with c1:
            st.image(img, caption="Uploaded image", use_container_width=True)
        with c2:
            demo_result = {
                "Tomato":("Early Blight", 91.4, "Moderate"),
                "Potato":("Late Blight", 88.2, "High"),
                "Cotton":("Leaf Curl Virus", 79.6, "Moderate"),
                "Wheat":("Healthy", 96.1, "None"),
                "Rice":("Leaf Blast", 84.3, "Moderate"),
                "Grapes":("Powdery Mildew", 87.0, "Low"),
            }[crop_choice]
            disease, conf, severity = demo_result
            st.markdown(f"""
            <div class="agx-result">
                <div style="font-size:20px; font-weight:800; color:#1B5E20;">{disease}</div>
                <div style="font-size:13px; color:#558B2F;">Confidence: {conf}% · Severity: {severity}</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.markdown("**Recommended action**")
            if disease == "Healthy":
                st.success("No treatment needed. Continue regular monitoring.")
            else:
                st.warning(f"Apply recommended fungicide per label dosage for {disease.lower()}. Avoid spraying if rain is forecast within 24 hrs — check the Weather tab.")
    else:
        st.info("Ek leaf image upload kar demo result baghण्यासाठी.")

# ---------------- TAB 3: YIELD PREDICTION (DEMO) ----------------
with tabs[2]:
    st.markdown('#### Yield Prediction <span class="agx-badge badge-demo">DEMO</span>', unsafe_allow_html=True)
    st.caption("Estimates expected yield, revenue, cost and profit from farm history and current conditions.")
    c1,c2,c3 = st.columns(3)
    with c1:
        y_crop = st.selectbox("Crop", list(CROP_ICONS.keys()), index=0)
        y_area = st.number_input("Farm area (acres)", 0.1, 100.0, land_area)
    with c2:
        y_irrigation = st.selectbox("Irrigation", ["Rain-fed","Drip","Canal","Borewell"])
        y_fert = st.selectbox("Fertilizer usage", ["Low","Medium","High"])
    with c3:
        y_prev_yield = st.number_input("Previous yield (kg/acre)", 0, 5000, 1200)
        y_cost = st.number_input("Farming cost so far (₹/acre)", 0, 100000, 18000)

    if st.button("📊 Estimate Yield", type="primary"):
        multiplier = {"Low":0.85,"Medium":1.0,"High":1.15}[y_fert]
        irr_bonus = {"Rain-fed":0.9,"Canal":1.0,"Drip":1.15,"Borewell":1.05}[y_irrigation]
        expected_yield = y_prev_yield * multiplier * irr_bonus * y_area
        price_per_kg = 22
        revenue = expected_yield * price_per_kg
        cost = y_cost * y_area
        profit = revenue - cost

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Expected yield", f"{expected_yield:,.0f} kg")
        m2.metric("Est. revenue", f"₹{revenue:,.0f}")
        m3.metric("Est. cost", f"₹{cost:,.0f}")
        m4.metric("Potential profit", f"₹{profit:,.0f}", delta=f"{(profit/cost*100 if cost else 0):.1f}% margin")
        st.caption("Estimates only — actual results depend on weather, pests and market conditions.")

# ---------------- TAB 4: PRICE FORECAST (DEMO) ----------------
with tabs[3]:
    st.markdown('#### Market Price Forecast <span class="agx-badge badge-demo">DEMO</span>', unsafe_allow_html=True)
    st.caption("Historical mandi price trend + short-term forecast with uncertainty band.")
    pr_crop = st.selectbox("Select crop", list(CROP_ICONS.keys()), key="price_crop")

    rng = np.random.default_rng(abs(hash(pr_crop)) % 1000)
    base = 1800 + (abs(hash(pr_crop)) % 1500)
    days = pd.date_range(end=datetime.today(), periods=30)
    hist = base + np.cumsum(rng.normal(0, 25, 30))
    future_days = pd.date_range(start=datetime.today()+timedelta(days=1), periods=10)
    forecast = hist[-1] + np.cumsum(rng.normal(2, 20, 10))
    upper = forecast + np.linspace(20, 120, 10)
    lower = forecast - np.linspace(20, 120, 10)

    df = pd.DataFrame({
        "date": list(days) + list(future_days),
        "price": list(hist) + list(forecast),
        "type": ["Historical"]*30 + ["Forecast"]*10
    })
    st.line_chart(df.set_index("date")["price"])
    c1,c2,c3 = st.columns(3)
    c1.metric("Current modal price", f"₹{hist[-1]:,.0f}/quintal")
    c2.metric("10-day forecast", f"₹{forecast[-1]:,.0f}/quintal", delta=f"{forecast[-1]-hist[-1]:+.0f}")
    c3.metric("Forecast range", f"₹{lower[-1]:,.0f} – ₹{upper[-1]:,.0f}")
    st.caption("Shown as a range with uncertainty — never a guaranteed future price.")

# ---------------- TAB 5: WEATHER INTELLIGENCE (DEMO) ----------------
with tabs[4]:
    st.markdown('#### Weather Intelligence <span class="agx-badge badge-demo">DEMO</span>', unsafe_allow_html=True)
    st.caption(f"Forecast for {location}, combined with your farm activity to give a decision — not just a number.")
    rng = np.random.default_rng(7)
    days7 = pd.date_range(start=datetime.today(), periods=7)
    rain_prob = rng.integers(5, 95, 7)
    temps = rng.integers(22, 36, 7)

    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="agx-card" style="text-align:center; padding:12px;">
                <div style="font-size:12px; color:#777;">{days7[i].strftime('%a')}</div>
                <div style="font-size:22px;">{'🌧️' if rain_prob[i]>55 else '⛅' if rain_prob[i]>25 else '☀️'}</div>
                <div style="font-size:13px; font-weight:700;">{temps[i]}°C</div>
                <div style="font-size:11px; color:#2E7D32;">{rain_prob[i]}% rain</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    high_rain_days = [days7[i].strftime('%A') for i in range(7) if rain_prob[i] > 55]
    if high_rain_days:
        st.warning(f"⚠️ High rain probability on **{', '.join(high_rain_days)}** — consider postponing spraying or fertilizer application on those days.")
    else:
        st.success("✅ No high-rain days in the next 7 days — a good window for spraying or harvest activity.")

# ---------------- TAB 6: SOIL ANALYSIS (DEMO) ----------------
with tabs[5]:
    st.markdown('#### Soil Intelligence <span class="agx-badge badge-demo">DEMO</span>', unsafe_allow_html=True)
    st.caption("Upload a soil test report (or enter values) to get a fertilizer and crop-suitability read.")
    s1,s2 = st.columns(2)
    with s1:
        st.file_uploader("Upload soil report (image/PDF)", type=["jpg","png","pdf"], key="soil_upload")
        st.caption("OCR extraction shown below is illustrative — connect a real OCR pipeline for production.")
    with s2:
        soil_ph = st.number_input("pH", 0.0, 14.0, 6.4)
        soil_n = st.number_input("Nitrogen (kg/ha)", 0, 500, 240)
        soil_oc = st.number_input("Organic Carbon (%)", 0.0, 5.0, 0.6)

    if st.button("🧪 Analyze Soil"):
        notes = []
        if soil_ph < 5.5: notes.append("Soil is acidic — consider liming before the next crop cycle.")
        elif soil_ph > 7.5: notes.append("Soil is alkaline — gypsum application may help.")
        else: notes.append("pH is in a healthy range for most crops.")
        if soil_oc < 0.5: notes.append("Organic carbon is low — add compost or green manure.")
        else: notes.append("Organic carbon level is adequate.")
        if soil_n < 200: notes.append("Nitrogen is on the lower side — a split urea application is advisable.")
        else: notes.append("Nitrogen level is sufficient for most cereal crops.")

        st.markdown('<div class="agx-result">', unsafe_allow_html=True)
        for note in notes:
            st.write("• " + note)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 7: FARMER ASSISTANT (DEMO CHAT) ----------------
# ---------------- TAB 7: FARMER ASSISTANT (LIVE - GROQ, multilingual, reply-to) ----------------
with tabs[6]:
    st.markdown('#### AI Farmer Assistant <span class="agx-badge badge-live">LIVE</span>', unsafe_allow_html=True)
    st.caption("Powered by Groq (Llama 3.1) — Marathi, Hindi ani English madhe bolू शकता.")

    import uuid

    client = groq_client  # shared client set up once near the top of the file
    if not groq_ready:
        st.warning("GROQ_API_KEY sapडली nahi — .streamlit/secrets.toml madhe takar Streamlit Cloud settings madhe add kar.")

    lang = st.radio(
        "Language / भाषा",
        ["मराठी (Marathi)", "हिंदी (Hindi)", "English"],
        horizontal=True
    )
    lang_instruction = {
        "मराठी (Marathi)": "फक्त मराठी भाषेत उत्तर दे. Devanagari script वापर.",
        "हिंदी (Hindi)": "सिर्फ हिंदी भाषा में जवाब दो। Devanagari script इस्तेमाल करो।",
        "English": "Reply only in plain English."
    }[lang]
    greetings = {
        "मराठी (Marathi)": "नमस्कार! तुमच्या पीक, रोग, हवामान किंवा माती याबद्दल काहीही विचारा.",
        "हिंदी (Hindi)": "नमस्ते! अपनी फसल, बीमारी, मौसम या मिट्टी के बारे में कुछ भी पूछें।",
        "English": "Hello! Ask me anything about your crop, disease, weather or soil."
    }

    if "chat" not in st.session_state or st.session_state.get("chat_lang") != lang:
        st.session_state.chat = [{"id": str(uuid.uuid4()), "role":"assistant", "text": greetings[lang], "reply_to": None}]
        st.session_state.chat_lang = lang
    if "replying_to" not in st.session_state:
        st.session_state.replying_to = None

    def find_msg(msg_id):
        for m in st.session_state.chat:
            if m["id"] == msg_id:
                return m
        return None

    def snippet(text, n=60):
        return text if len(text) <= n else text[:n] + "..."

    # ---- Render chat history ----
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            if msg.get("reply_to"):
                original = find_msg(msg["reply_to"])
                if original:
                    st.markdown(
                        f"""<div style="border-left:3px solid #A5D6A7; background:#F1F8F2;
                        padding:6px 10px; border-radius:6px; font-size:12.5px; color:#557a5c; margin-bottom:6px;">
                        ↪ {'You' if original['role']=='user' else 'Assistant'}: {snippet(original['text'])}
                        </div>""",
                        unsafe_allow_html=True
                    )
            st.write(msg["text"])
            if st.button("↩ Reply", key=f"reply_{msg['id']}"):
                st.session_state.replying_to = msg["id"]
                st.rerun()

    # ---- Reply preview above input ----
    if st.session_state.replying_to:
        original = find_msg(st.session_state.replying_to)
        if original:
            rc1, rc2 = st.columns([10,1])
            with rc1:
                st.markdown(
                    f"""<div style="border-left:3px solid #66BB6A; background:#E8F5E9;
                    padding:8px 12px; border-radius:6px; font-size:13px; color:#2E5D34;">
                    Replying to: {snippet(original['text'], 80)}
                    </div>""",
                    unsafe_allow_html=True
                )
            with rc2:
                if st.button("✕", key="cancel_reply"):
                    st.session_state.replying_to = None
                    st.rerun()

    placeholder_text = {
        "मराठी (Marathi)": "तुमचा प्रश्न इथे लिहा...",
        "हिंदी (Hindi)": "अपना सवाल यहाँ लिखें...",
        "English": "Type your question..."
    }[lang]

    user_q = st.chat_input(placeholder_text)
    if user_q and groq_ready:
        reply_ref = st.session_state.replying_to
        user_msg = {"id": str(uuid.uuid4()), "role":"user", "text": user_q, "reply_to": reply_ref}
        st.session_state.chat.append(user_msg)
        st.session_state.replying_to = None

        # build context — include the replied-to message explicitly if present
        context_note = ""
        if reply_ref:
            original = find_msg(reply_ref)
            if original:
                context_note = f"\n\n(Farmer is specifically replying to this earlier message: \"{original['text']}\")"

        system_prompt = f"""Tu AGRINEX AI cha farming assistant ahes. Farmer profile:
Name: {farmer_name}, Location: {location}, Land: {land_area} acres.

{lang_instruction}

Farming, crops, disease, soil, weather, market price संबंधित प्रश्नांना उत्तर दे — short, practical, farmer-friendly. Farming shivay dusrya topic var answer dyaycha nahi, politely redirect kar.{context_note}"""

        history = [{"role":m["role"],"content":m["text"]} for m in st.session_state.chat[-6:]]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":system_prompt}, *history],
            temperature=0.6,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        assistant_msg = {"id": str(uuid.uuid4()), "role":"assistant", "text": answer, "reply_to": None}
        st.session_state.chat.append(assistant_msg)
        st.rerun()

# ---------------- TAB 8: MARKETPLACE (DEMO) ----------------
with tabs[7]:
    st.markdown('#### Marketplace <span class="agx-badge badge-demo">DEMO</span>', unsafe_allow_html=True)
    m1, m2 = st.tabs(["🌾 Sell your crop", "🧑‍🌾 Farm labor"])

    with m1:
        st.caption("List your crop for buyers to discover.")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            list_crop = st.selectbox("Crop", list(CROP_ICONS.keys()), key="list_crop")
            list_qty = st.number_input("Quantity (kg)", 0, 50000, 2000)
        with cc2:
            list_price = st.number_input("Expected price (₹/kg)", 0, 500, 24)
            list_harvest = st.date_input("Expected harvest date")
        with cc3:
            list_quality = st.selectbox("Quality grade", ["A - Premium","B - Standard","C - Basic"])
        if st.button("📋 Create listing"):
            st.success(f"Listed: {list_qty} kg of {list_crop} at ₹{list_price}/kg, grade {list_quality}, ready {list_harvest.strftime('%d %b %Y')}. Buyers near {location} will see this listing.")

    with m2:
        st.caption("Post a farm job for local workers.")
        jc1, jc2 = st.columns(2)
        with jc1:
            job_type = st.selectbox("Job type", ["Harvesting","Sowing","Weeding","Spraying","General labor"])
            workers = st.number_input("Workers needed", 1, 50, 5)
        with jc2:
            days_needed = st.number_input("Duration (days)", 1, 60, 2)
            wage = st.number_input("Wage per worker/day (₹)", 100, 2000, 400)
        if st.button("📋 Post job"):
            st.success(f"Job posted: {workers} workers for {job_type}, {days_needed} day(s), ₹{wage}/day at {location}.")

st.write("")
st.divider()
st.caption("AGRINEX AI — prototype. Crop Recommendation tab uses the real trained model when crop_model.pkl and crop_labels.json are present in this folder; other tabs are illustrative UI for the full planned system.")
