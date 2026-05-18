"""Header component for authenticated dashboard pages."""
import streamlit as st
from . import auth


def render_header():
    """Render a top header bar with user email and logout button."""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("### 🚗 ARTES Dashboard")
    
    with col2:
        st.markdown("Système de prévision des ventes")
    
    with col3:
        email = st.session_state.get("auth_email", "")
        if email:
            st.markdown(f"**{email}**", unsafe_allow_html=True)
            if st.button("Se déconnecter", key="header_logout", use_container_width=True):
                auth.logout()
    
    st.markdown("---")


def render_sidebar_minimal():
    """Render a minimal sidebar (just branding, no logout)."""
    st.sidebar.markdown("### ARTES")
    st.sidebar.markdown("Prévision des ventes")
