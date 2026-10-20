import streamlit as st

from firebase_auth import reset_password, sign_in, sign_up
from firestore_db import get_profile, save_profile


def _finish_login(user):
    try:
        user["profile"] = get_profile(user["uid"])
    except Exception:
        user["profile"] = {}
        st.toast("Profile load झाली नाही (Firestore तपासा)", icon="⚠️")
    st.session_state["user"] = user
    st.rerun()


def require_login():
    """app.py च्या सुरुवातीला call कर. Login नसेल तर form दाखवून stop करतो."""
    if "user" in st.session_state:
        return st.session_state["user"]

    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown(
            '<div class="agx-login-brand"><div class="logo">🌾</div>'
            '<div class="name">AGRINEX AI</div>'
            '<div class="tag">शेतकऱ्यांसाठी स्मार्ट शेती सल्लागार</div></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign up"])

            with tab_login:
                with st.form("login_form"):
                    email = st.text_input("Email", placeholder="you@example.com")
                    pw = st.text_input("Password", type="password")
                    go = st.form_submit_button("Login", type="primary", use_container_width=True)
                if go:
                    user = None
                    if not email or not pw:
                        st.error("Email आणि password दोन्ही टाक.")
                    else:
                        with st.spinner("Login होतंय..."):
                            try:
                                user = sign_in(email.strip().lower(), pw)
                            except ValueError as e:
                                st.error(str(e))
                    if user:
                        _finish_login(user)

                with st.expander("Password विसरलास?"):
                    reset_email = st.text_input("Registered email", key="reset_email")
                    if st.button("Reset link पाठवा", key="reset_btn"):
                        if reset_email.strip():
                            reset_password(reset_email.strip().lower())
                            st.success("Reset link email वर पाठवली (Spam folder पण बघ).")
                        else:
                            st.warning("आधी email टाक.")

            with tab_signup:
                with st.form("signup_form"):
                    name = st.text_input("पूर्ण नाव")
                    place = st.text_input("गाव, जिल्हा", placeholder="उदा. Nashik, Maharashtra")
                    email2 = st.text_input("Email", key="su_email", placeholder="you@example.com")
                    pw2 = st.text_input("Password (किमान 6 अक्षरे)", type="password", key="su_pw")
                    pw3 = st.text_input("Password पुन्हा टाक", type="password", key="su_pw2")
                    go2 = st.form_submit_button("Account बनवा", type="primary", use_container_width=True)
                if go2:
                    if not name.strip() or not email2.strip():
                        st.error("नाव आणि email आवश्यक आहे.")
                    elif len(pw2) < 6:
                        st.error("Password किमान 6 अक्षरांचा हवा.")
                    elif pw2 != pw3:
                        st.error("दोन्ही password सारखे नाहीत.")
                    else:
                        new_user = None
                        with st.spinner("Account बनतंय..."):
                            try:
                                new_user = sign_up(email2.strip().lower(), pw2, name.strip())
                            except ValueError as e:
                                st.error(str(e))
                        if new_user:
                            profile = {
                                "name": name.strip(),
                                "location": place.strip(),
                                "email": new_user["email"],
                            }
                            try:
                                save_profile(new_user["uid"], profile)
                            except Exception as e:
                                st.toast(f"Account बनलं, पण profile save नाही झाली: {e}", icon="⚠️")
                            new_user["profile"] = profile
                            st.session_state["user"] = new_user
                            st.rerun()

    st.stop()


def logout_button():
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
