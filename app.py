import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import cv2
from PIL import Image
from mandi_data import (
    INDIAN_STATES, fetch_state_snapshot, summarize_by_commodity,
    log_daily_snapshot, get_commodity_history, naive_forecast,
)

DISEASE_MODEL_PATHS = ["disease_model.keras", os.path.join("model_output", "disease_model.h5")]
DISEASE_LABELS_PATHS = ["disease_labels.json", os.path.join("model_output", "disease_labels.json")]
DISEASE_INFO_PATH = "disease_info.json"

disease_model = None
disease_labels = []
disease_info = {}
disease_model_is_real = False

disease_model_path = next((p for p in DISEASE_MODEL_PATHS if os.path.exists(p)), None)
disease_labels_path = next((p for p in DISEASE_LABELS_PATHS if os.path.exists(p)), None)

if disease_model_path and disease_labels_path:
    try:
        import tensorflow as tf
        disease_model = tf.keras.models.load_model(disease_model_path)
        with open(disease_labels_path) as f:
            disease_labels = json.load(f)
        if os.path.exists(DISEASE_INFO_PATH):
            with open(DISEASE_INFO_PATH) as f:
                disease_info = json.load(f)
        disease_model_is_real = True
    except Exception as e:
        st.sidebar.warning(f"Disease model found but couldn't load: {e}")

def preprocess_leaf_image(uploaded_file, img_size=224):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(uploaded_file):
    img_array = preprocess_leaf_image(uploaded_file)
    preds = disease_model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    label = disease_labels[top_idx]
    confidence = float(preds[top_idx]) * 100
    info = disease_info.get(label, {"severity": "Unknown", "action": "Consult a local agriculture officer for exact treatment."})
    return label, confidence, info

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

data_gov_key = st.secrets.get("DATA_GOV_API_KEY", None)

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

# ---------------- TAB 2: DISEASE DETECTION (LIVE - CNN + OpenCV) ----------------
with tabs[1]:
    badge = '<span class="agx-badge badge-live">LIVE MODEL</span>' if disease_model_is_real else '<span class="agx-badge badge-demo">DEMO MODE</span>'
    st.markdown(f"#### Crop Disease Detection {badge}", unsafe_allow_html=True)
    st.caption("Upload a leaf photo — MobileNetV2 CNN trained on plant-disease imagery, preprocessed with OpenCV.")

    img = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"], key="disease_upload")

    if img:
        c1, c2 = st.columns([1,1])
        with c1:
            st.image(img, caption="Uploaded image", use_container_width=True)
            img.seek(0)

        with c2:
            if disease_model_is_real:
                with st.spinner("Analyzing leaf image..."):
                    label, conf, info = predict_disease(img)
                disease_display = label.replace("_", " ")
                st.markdown(f"""
                <div class="agx-result">
                    <div style="font-size:20px; font-weight:800; color:#1B5E20;">{disease_display}</div>
                    <div style="font-size:13px; color:#558B2F;">Confidence: {conf:.1f}% · Severity: {info['severity']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.markdown("**Recommended action**")
                if info["severity"] == "None":
                    st.success(info["action"])
                elif info["severity"] == "High":
                    st.error(info["action"] + " Check the Weather tab before spraying.")
                else:
                    st.warning(info["action"] + " Avoid spraying if rain is forecast within 24 hrs — check the Weather tab.")
            else:
                crop_choice = st.selectbox("Crop", ["Tomato","Potato","Cotton","Wheat","Rice","Grapes"], key="demo_crop_choice")
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
        st.info("Ek leaf image upload kar result baghण्यासाठी.")

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

# ---------------- TAB 4: PRICE FORECAST — live state-wide market board ----------------
with tabs[3]:
    badge = '<span class="agx-badge badge-live">LIVE DATA</span>' if data_gov_key else '<span class="agx-badge badge-demo">DEMO MODE</span>'
    st.markdown(f"#### Market Price Board {badge}", unsafe_allow_html=True)
    st.caption("Pick a state to see every crop's live mandi price at once — tap any crop below to open its own chart.")

    if not data_gov_key:
        st.warning("DATA_GOV_API_KEY sapडली nahi — .streamlit/secrets.toml madhe takar Streamlit Cloud settings madhe add kar.")
    else:
        selected_state = st.selectbox("State", ["All India"] + INDIAN_STATES, key="price_state")
        state_filter = None if selected_state == "All India" else selected_state

        if st.session_state.get("_price_state_prev") != selected_state:
            st.session_state["_price_state_prev"] = selected_state
            st.session_state["_selected_commodity"] = None

        with st.spinner(f"Loading live prices for {selected_state}..."):
            raw_df, status = fetch_state_snapshot(data_gov_key, state=state_filter)

        # TEMPORARY DEBUG: show the exact failure reason instead of a generic message.
        # Remove this block once the root cause is confirmed fixed.
        if status is not True:
            st.error(f"Debug — live fetch failed: {status}")

        board = summarize_by_commodity(raw_df)

        if status is not True or board.empty:
            cached = st.session_state.get(f"_board_cache_{selected_state}")
            if cached is not None and not cached.empty:
                board = cached
                st.caption("🟡 Live refresh failed — showing the last loaded prices for this state.")
            else:
                st.info(f"No live price data available for {selected_state} right now. See the debug message above for the exact reason.")
        else:
            st.session_state[f"_board_cache_{selected_state}"] = board
            log_daily_snapshot(selected_state, board)

        if not board.empty:
            icon_of = lambda name: CROP_ICONS.get(name.strip().lower(), "🌱")

            st.write("")
            st.caption(f"🟢 {len(board)} crops trading in {selected_state} today · tap a row to open its chart")

            display_df = board.copy()
            display_df.insert(0, "", display_df["commodity"].apply(icon_of))
            display_df = display_df.rename(columns={
                "commodity": "Crop", "avg_modal": "Avg Price (₹/quintal)",
                "min_price": "Low", "max_price": "High", "markets": "Markets",
            })
            display_df["Avg Price (₹/quintal)"] = display_df["Avg Price (₹/quintal)"].round(0)

            selected_row = None
            try:
                event = st.dataframe(
                    display_df, use_container_width=True, hide_index=True,
                    on_select="rerun", selection_mode="single-row", key="price_board_table",
                )
                rows = event.selection.rows if event and event.selection else []
                if rows:
                    selected_row = board.iloc[rows[0]]["commodity"]
            except TypeError:
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            if selected_row:
                st.session_state["_selected_commodity"] = selected_row

            crop_names = board["commodity"].tolist()
            current = st.session_state.get("_selected_commodity")
            default_idx = crop_names.index(current) + 1 if current in crop_names else 0
            picked = st.selectbox(
                "Or choose a crop to view", ["— select —"] + crop_names,
                index=default_idx, key="crop_picker",
            )
            if picked != "— select —":
                st.session_state["_selected_commodity"] = picked

            focus = st.session_state.get("_selected_commodity")

            if focus:
                row = board[board["commodity"] == focus].iloc[0]
                st.write("")
                st.markdown(f"### {icon_of(focus)} {focus}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Avg modal price", f"₹{row['avg_modal']:,.0f}/quintal")
                m2.metric("Lowest reported", f"₹{row['min_price']:,.0f}")
                m3.metric("Highest reported", f"₹{row['max_price']:,.0f}")
                m4.metric("Markets reporting", f"{int(row['markets'])}")

                crop_df = raw_df[raw_df["commodity"] == focus]
                if "market" in crop_df.columns and not crop_df.empty:
                    by_market = crop_df.groupby("market", as_index=False)["modal_price"].mean().sort_values("modal_price")
                    st.caption("Today's price across markets")
                    st.bar_chart(by_market.set_index("market")["modal_price"])

                hist = get_commodity_history(selected_state, focus)
                st.caption("Price trend (built from this app's own daily visits — the source data is a same-day snapshot with no built-in history)")
                if len(hist) >= 2:
                    forecast_vals = naive_forecast(hist, days_ahead=3)
                    chart_df = hist[["date", "avg_modal"]].rename(columns={"avg_modal": "Recorded"}).set_index("date")
                    if forecast_vals is not None:
                        last_date = pd.to_datetime(hist["date"].iloc[-1])
                        future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(len(forecast_vals))]
                        proj_df = pd.DataFrame({"date": future_dates, "Projected": forecast_vals}).set_index("date")
                        combo = pd.concat([chart_df, proj_df], axis=0).sort_index()
                        st.line_chart(combo)
                        st.caption(f"Projected next price (naive trend, {len(hist)} day(s) of history so far): ₹{forecast_vals[0]:,.0f}/quintal. Treat as a rough direction, not a guarantee.")
                    else:
                        st.line_chart(chart_df)
                else:
                    st.info("Only today's price is logged so far for this crop — come back on a future day to start seeing a trend and projection build up.")

        st.caption("Source: data.gov.in (Agmarknet), daily arrivals snapshot.")

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

# ---------------- TAB 7: FARMER ASSISTANT (LIVE - GROQ, multilingual, reply-to) ----------------
with tabs[6]:
    st.markdown('#### AI Farmer Assistant <span class="agx-badge badge-live">LIVE</span>', unsafe_allow_html=True)
    st.caption("Powered by Groq (Llama 3.1) — Marathi, Hindi ani English madhe bolू शकता.")

    import uuid

    client = groq_client
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
            model="openai/gpt-oss-20b",
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
st.caption("AGRINEX AI — prototype. Crop Recommendation, Disease Detection & Price Forecast tabs use real live data/models when their files/keys are present; other tabs are illustrative UI for the full planned system.")
