"""Firestore access via firebase-admin. Data lives under users/{uid}/..."""

from datetime import datetime, timedelta, timezone

import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore

IST = timezone(timedelta(hours=5, minutes=30))


@st.cache_resource
def _db():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_service_account"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def _user_doc(uid):
    return _db().collection("users").document(uid)


def get_profile(uid):
    snap = _user_doc(uid).get()
    return snap.to_dict() or {} if snap.exists else {}


def save_profile(uid, data: dict):
    _user_doc(uid).set({**data, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)


def add_record(uid, collection, data: dict):
    _user_doc(uid).collection(collection).add({**data, "created_at": firestore.SERVER_TIMESTAMP})


def list_records(uid, collection, limit=30):
    q = (
        _user_doc(uid)
        .collection(collection)
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    rows = []
    for d in q.stream():
        row = d.to_dict() or {}
        row["id"] = d.id
        ts = row.get("created_at")
        if ts is not None:
            row["created_at"] = datetime.fromtimestamp(ts.timestamp(), IST)
        rows.append(row)
    return rows


def safe_add_record(uid, collection, data: dict):
    """Save करताना error आला तरी app crash होत नाही."""
    try:
        add_record(uid, collection, data)
        return True
    except Exception as e:
        st.toast(f"Save नाही झालं: {e}", icon="⚠️")
        return False
