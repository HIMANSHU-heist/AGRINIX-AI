"""Firebase Authentication (Email/Password) via REST API. Server-side only."""

import requests
import streamlit as st

_BASE = "https://identitytoolkit.googleapis.com/v1"

_MESSAGES = {
    "EMAIL_EXISTS": "हा email आधीच registered आहे. Login कर.",
    "INVALID_LOGIN_CREDENTIALS": "Email किंवा password चुकीचा आहे.",
    "INVALID_PASSWORD": "Email किंवा password चुकीचा आहे.",
    "EMAIL_NOT_FOUND": "Email किंवा password चुकीचा आहे.",
    "INVALID_EMAIL": "Email format बरोबर नाही.",
    "WEAK_PASSWORD": "Password किमान 6 अक्षरांचा हवा.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "खूप वेळा प्रयत्न झाले. थोड्या वेळाने पुन्हा try कर.",
    "USER_DISABLED": "हे account बंद केलेलं आहे.",
    "OPERATION_NOT_ALLOWED": "Firebase Console मध्ये Authentication > Sign-in method > Email/Password enable कर.",
}


def _post(endpoint, payload):
    try:
        r = requests.post(
            f"{_BASE}/{endpoint}",
            params={"key": st.secrets["firebase"]["apiKey"]},
            json=payload,
            timeout=15,
        )
    except requests.RequestException:
        raise ValueError("Network error. थोड्या वेळाने पुन्हा try कर.")
    try:
        data = r.json()
    except ValueError:
        raise ValueError("Auth service ने अनपेक्षित उत्तर दिलं.")
    if r.status_code != 200:
        code = str(data.get("error", {}).get("message", "UNKNOWN")).split(" ")[0]
        raise ValueError(_MESSAGES.get(code, f"Auth error: {code}"))
    return data


def _user_from(d, email, name=""):
    return {
        "uid": d["localId"],
        "email": d.get("email", email),
        "name": d.get("displayName") or name,
        "id_token": d.get("idToken", ""),
        "refresh_token": d.get("refreshToken", ""),
    }


def sign_in(email, password):
    d = _post("accounts:signInWithPassword", {"email": email, "password": password, "returnSecureToken": True})
    return _user_from(d, email)


def sign_up(email, password, name=""):
    d = _post("accounts:signUp", {"email": email, "password": password, "returnSecureToken": True})
    if name:
        try:
            _post("accounts:update", {"idToken": d["idToken"], "displayName": name, "returnSecureToken": False})
        except ValueError:
            pass
    return _user_from(d, email, name)


def reset_password(email):
    try:
        _post("accounts:sendOobCode", {"requestType": "PASSWORD_RESET", "email": email})
    except ValueError:
        pass  # email exist करतो का हे उघड करायचं नाही
