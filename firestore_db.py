

def safe_add_record(uid, collection, data: dict):
    """Save करताना error आला तरी app crash होत नाही."""
    try:
        add_record(uid, collection, data)
        return True
    except Exception as e:
        st.toast(f"Save नाही झालं: {e}", icon="⚠️")
        return False
