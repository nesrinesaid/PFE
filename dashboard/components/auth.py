import os
import random
import re
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 30))
CODE_VALIDITY_MINUTES = int(os.getenv("CODE_VALIDITY_MINUTES", 10))
EMAIL_REGEX = re.compile(r"^(?:[\w\.-]+@(?:artes\.com\.tn|renault\.com\.tn)|nesrine\.said@etudiant-fsegt\.utm\.tn)$", re.IGNORECASE)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #E9EBF7; }
[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E5F0; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: #6B7490 !important; font-size: 13px; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1E2A45 !important; }
.metric-card { background: #FFFFFF; border-radius: 10px; border: 1px solid #E2E5F0; padding: 16px 20px; border-top: 3px solid #4A69A9; }
.metric-card.orange { border-top-color: #FF944B; }
.metric-card.green  { border-top-color: #2ECC8F; }
.metric-card.red    { border-top-color: #E84855; }
.metric-value { font-size: 30px; font-weight: 700; color: #1E2A45; line-height: 1.1; }
.metric-delta-pos { font-size: 12px; font-weight: 500; color: #2ECC8F; }
.metric-delta-neg { font-size: 12px; font-weight: 500; color: #E84855; }
.metric-label { font-size: 12px; color: #6B7490; margin-top: 4px; font-weight: 400; }
.chart-panel { background: #FFFFFF; border-radius: 10px; border: 1px solid #E2E5F0; padding: 20px; }
.section-title { font-size: 16px; font-weight: 600; color: #1E2A45; margin: 0 0 4px 0; }
.section-subtitle { font-size: 12px; color: #9BA3B8; margin: 0 0 16px 0; }
.stButton > button { background-color: #4A69A9; color: white; border: none; border-radius: 6px; padding: 8px 20px; font-weight: 500; font-family: 'Inter', sans-serif; font-size: 13px; transition: background-color 0.15s; box-shadow: none; }
.stButton > button:hover { background-color: #2E4A80; box-shadow: none; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; }
.badge-blue   { background: #EEF2FF; color: #4A69A9; }
.badge-orange { background: #FFF3EB; color: #FF944B; }
.badge-green  { background: #EDFAF4; color: #2ECC8F; }
.badge-red    { background: #FDECEA; color: #E84855; }
.stDataFrame { border-radius: 8px; overflow: hidden; }
hr { border: none; border-top: 1px solid #E2E5F0; margin: 16px 0; }
.block-container { padding-top: 1.25rem; padding-bottom: 1rem; }
div[data-testid="stAlert"] { border-radius: 8px; }
.dashboard-footer { text-align: center; color: #9BA3B8; font-size: 11px; padding: 16px 0 8px; border-top: 1px solid #E2E5F0; margin-top: 24px; }
.login-card { background: #FFFFFF; border-radius: 16px; border: 1px solid #E2E5F0; padding: 32px 28px; margin-top: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.login-logo { display: flex; justify-content: center; margin-bottom: 16px; }
.login-title { text-align: center; font-weight: 700; color: #1E2A45; margin-bottom: 8px; font-size: 22px; }
.login-subtitle { text-align: center; color: #6B7490; font-size: 13px; margin-bottom: 24px; }
.login-hint { color: #9BA3B8; font-size: 12px; text-align: center; }
</style>
"""


def apply_global_styles():
    st.markdown(CSS, unsafe_allow_html=True)


def _init_auth_state():
    defaults = {
        "authenticated": False,
        "login_sent": False,
        "auth_email": "",
        "auth_code": "",
        "auth_code_ts": 0.0,
        "last_activity": 0.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _logo_path():
    candidates = [ROOT_DIR / "assets" / "logo_artes.png", ROOT_DIR / "dashboard" / "assets" / "logo_artes.png"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def is_email_valid(email: str) -> bool:
    return bool(EMAIL_REGEX.match((email or "").strip()))


def is_authenticated() -> bool:
    _init_auth_state()
    return bool(st.session_state.get("authenticated"))


def touch_activity():
    if st.session_state.get("authenticated"):
        st.session_state["last_activity"] = time.time()


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def check_session_timeout():
    _init_auth_state()
    if not st.session_state.get("authenticated"):
        return
    last = float(st.session_state.get("last_activity", 0.0) or 0.0)
    if last and (time.time() - last) > SESSION_TIMEOUT_MINUTES * 60:
        st.warning("Session expirée pour cause d'inactivité. Merci de vous reconnecter.")
        logout()
    else:
        touch_activity()


def code_remaining_seconds() -> int:
    ts = float(st.session_state.get("auth_code_ts", 0.0) or 0.0)
    if not ts:
        return 0
    remaining = int(CODE_VALIDITY_MINUTES * 60 - (time.time() - ts))
    return max(0, remaining)


def _smtp_settings():
    return {
        "server": os.getenv("SMTP_SERVER", "").strip(),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "email": os.getenv("SMTP_EMAIL", "").strip(),
        "password": os.getenv("SMTP_PASSWORD", "").strip(),
        "use_ssl": os.getenv("SMTP_USE_SSL", "false").strip().lower() in {"1", "true", "yes", "on"},
        "use_starttls": os.getenv("SMTP_USE_STARTTLS", "true").strip().lower() in {"1", "true", "yes", "on"},
        "from_name": os.getenv("SMTP_FROM_NAME", "ARTES Dashboard").strip(),
    }


def send_verification_code(email: str) -> bool:
    code = f"{random.randint(0, 999999):06d}"
    st.session_state["auth_code"] = code
    st.session_state["auth_code_ts"] = time.time()
    st.session_state["login_sent"] = True
    st.session_state["auth_delivery_mode"] = "dev"

    smtp = _smtp_settings()
    subject = "Code de vérification — ARTES Dashboard"
    body = f"Votre code de vérification ARTES est : {code}\n\nValable {CODE_VALIDITY_MINUTES} minutes."

    if not all([smtp["server"], smtp["email"], smtp["password"]]):
        st.warning(
            "SMTP non configuré : l'envoi email est désactivé en mode développement. "
            "Le code ci-dessous peut être utilisé pour continuer."
        )
        st.code(code, language="text")
        return False

    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = f"{smtp['from_name']} <{smtp['email']}>" if smtp["from_name"] else smtp["email"]
        message["To"] = email
        message.set_content(body)

        if smtp["use_ssl"]:
            smtp_cls = smtplib.SMTP_SSL
        else:
            smtp_cls = smtplib.SMTP

        with smtp_cls(smtp["server"], smtp["port"], timeout=20) as server:
            if smtp["use_starttls"] and not smtp["use_ssl"]:
                server.starttls()
            server.login(smtp["email"], smtp["password"])
            server.send_message(message)
        st.session_state["auth_delivery_mode"] = "email"
        return True
    except Exception as exc:
        st.error(f"Échec de l'envoi du code : {exc}")
        st.session_state["auth_delivery_mode"] = "dev"
        st.info("Le code est disponible ci-dessous pour continuer en mode développement.")
        st.code(code, language="text")
        return False


def verify_code(input_code: str) -> bool:
    stored = str(st.session_state.get("auth_code", "")).strip()
    if not stored or code_remaining_seconds() <= 0:
        return False
    return str(input_code or "").strip() == stored


def render_login_page():
    apply_global_styles()
    _init_auth_state()
    
    # Create centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        logo = _logo_path()
        if logo:
            col_logo = st.columns([1])[0]
            with col_logo:
                st.image(logo, width=200, use_container_width=False)
        
        st.markdown('<h1 style="text-align: center; font-size: 24px; margin: 20px 0 8px;">Connexion sécurisée ARTES</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #6B7490; font-size: 14px; margin-bottom: 24px;">Accès réservé aux collaborateurs ARTES et Renault Tunisie</p>', unsafe_allow_html=True)

        email = st.text_input(
            "📧 Adresse email professionnelle",
            placeholder="prenom.nom@artes.com.tn",
            label_visibility="collapsed"
        )
        
        col_btn1 = st.columns([1])[0]
        with col_btn1:
            if st.button("🔓 Envoyer le code de vérification", use_container_width=True, key="send_code_btn"):
                if not is_email_valid(email):
                    st.error("❌ Accès réservé aux collaborateurs ARTES et Renault Tunisie.")
                else:
                    if send_verification_code(email):
                        st.session_state["auth_email"] = email
                        st.success("✅ Code envoyé. Vérifiez votre boîte email.")

        if st.session_state.get("login_sent"):
            remaining = code_remaining_seconds()
            st.markdown(
                f'<p style="text-align: center; color: #9BA3B8; font-size: 12px; margin: 12px 0;">Code valable : <strong>{remaining // 60:02d}:{remaining % 60:02d}</strong></p>',
                unsafe_allow_html=True,
            )
            default_code = st.session_state.get("auth_code", "") if st.session_state.get("auth_delivery_mode") == "dev" else ""
            code_input = st.text_input(
                "🔐 Code reçu par email",
                max_chars=6,
                placeholder="123456",
                value=default_code,
                label_visibility="collapsed"
            )
            col_btn2 = st.columns([1])[0]
            with col_btn2:
                if st.button("✓ Valider le code", use_container_width=True, key="verify_code_btn"):
                    if verify_code(code_input):
                        st.session_state["authenticated"] = True
                        st.session_state["last_activity"] = time.time()
                        st.session_state["login_sent"] = False
                        st.success("✅ Authentification réussie!")
                        st.rerun()
                    else:
                        if code_remaining_seconds() <= 0:
                            st.error("❌ Le code a expiré. Demandez-en un nouveau.")
                        else:
                            st.error("❌ Code invalide.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def require_auth():
    check_session_timeout()
    if not is_authenticated():
        render_login_page()
        st.stop()


def render_sidebar_header():
    st.sidebar.markdown("### ARTES Dashboard")
    st.sidebar.markdown("Système de prévision des ventes")


def render_logout_button():
    st.sidebar.markdown("---")
    if st.sidebar.button("Se déconnecter", use_container_width=True):
        logout()


def footer():
    st.markdown(
        '<div class="dashboard-footer">© 2026 ARTES — Système de Prévision des Ventes | Développé dans le cadre du PFE FSEGT</div>',
        unsafe_allow_html=True,
    )


def format_k(value) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def format_pct(value, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except Exception:
        return str(value)


def format_delta(delta_value, pct: bool = True) -> str:
    try:
        delta = float(delta_value)
    except Exception:
        return str(delta_value)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%" if pct else f"{sign}{delta:.0f}"


def metric_card_html(value: str, label: str, delta: str = "", kind: str = "primary") -> str:
    kind_class = {"orange": "orange", "green": "green", "red": "red"}.get(kind, "")
    if delta:
        delta_class = "metric-delta-pos" if str(delta).strip().startswith("+") else "metric-delta-neg"
        delta_html = f'<div class="{delta_class}">{delta}</div>'
    else:
        delta_html = ""
    return f"""
    <div class="metric-card {kind_class}">
        <div class="metric-value">{value}</div>
        {delta_html}
        <div class="metric-label">{label}</div>
    </div>
    """


def badge_html(text: str, kind: str = "blue") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def status_pill(status: str) -> str:
    mapping = {
        "success": ("badge-green", "✅"),
        "warning": ("badge-orange", "⚠️"),
        "danger": ("badge-red", "❌"),
        "info": ("badge-blue", "ℹ️"),
    }
    cls, icon = mapping.get(status, mapping["info"])
    return f'<span class="badge {cls}">{icon} {status.capitalize()}</span>'


def section_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)

