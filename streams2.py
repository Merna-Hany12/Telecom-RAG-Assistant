import streamlit as st
import requests
import pandas as pd
import uuid

# =========================
# 🔗 API & Sheet URLs
# =========================
API_URL   = "http://localhost:8000/ask"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1mO4xKCHFkOjlThKJQkH85DvSMMH969rXYDmtYeoZPBU/export?format=csv"

# =========================
#  PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NileTel Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# SESSION STATE INIT
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "text": ...,
    #              "needs_action": ..., "sources": [...], "ticket_problem": "..."}
    st.session_state.chat_history = []

# =========================
#  CUSTOM CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500&family=Cairo:wght@400;600;700&display=swap');

:root {
    --bg:        #05080f;
    --surface:   #0a0f1e;
    --surface2:  #0f1729;
    --surface3:  #141f38;
    --border:    #1a2d50;
    --border2:   #243d6b;
    --accent:    #00c8f0;
    --accent2:   #0044ff;
    --accent3:   #7b2fff;
    --warn:      #f5a623;
    --success:   #00e887;
    --danger:    #ff3d5a;
    --text:      #ddeaf8;
    --muted:     #3d5a7a;
    --muted2:    #5b7a9e;
    --glow-c:    0 0 32px rgba(0,200,240,.2);
    --glow-g:    0 0 32px rgba(0,232,135,.2);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 3rem 5rem; max-width: 1140px; }

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,68,255,.13), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(123,47,255,.08), transparent),
        var(--bg) !important;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,200,240,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,240,.025) 1px, transparent 1px);
    background-size: 56px 56px;
    pointer-events: none;
    z-index: 0;
}

.hero-wrap {
    position: relative;
    padding: 3.5rem 0 3rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    right: -80px; top: -80px;
    width: 420px; height: 420px;
    border-radius: 50%;
    border: 1px solid rgba(0,200,240,.08);
    pointer-events: none;
}
.hero-inner { display: flex; align-items: center; gap: 2.5rem; }
.hero-text  { flex: 1; }

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    font-size: .62rem;
    font-weight: 500;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(0,200,240,.07);
    border: 1px solid rgba(0,200,240,.18);
    border-radius: 2px;
    padding: .22rem .75rem;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 700;
    line-height: 1.04;
    letter-spacing: -.04em;
    background: linear-gradient(130deg, #ffffff 20%, var(--accent) 60%, var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .6rem;
}
.hero-sub {
    font-size: .78rem;
    color: var(--muted2);
    letter-spacing: .05em;
    display: flex;
    align-items: center;
    gap: .5rem;
    flex-wrap: wrap;
}
.hero-sub .sep { color: var(--border2); }

.signal-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--success);
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(0,232,135,.5); }
    50%      { opacity:.7; box-shadow: 0 0 0 7px rgba(0,232,135,0); }
}

.hero-illus {
    flex-shrink: 0;
    width: 180px;
    opacity: .85;
    filter: drop-shadow(0 0 24px rgba(0,200,240,.3));
    animation: float 6s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0px) rotate(-2deg); }
    50%      { transform: translateY(-10px) rotate(2deg); }
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .6rem;
    font-weight: 500;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .85rem;
    display: flex;
    align-items: center;
    gap: .6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color .25s, box-shadow .25s;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent), var(--accent3));
}
.card:hover {
    border-color: rgba(0,200,240,.3);
    box-shadow: 0 8px 40px rgba(0,0,0,.3);
}

/* ── Chat bubbles ── */
.bubble-wrap {
    display: flex;
    flex-direction: column;
    gap: .75rem;
    margin-bottom: 1.5rem;
}
.bubble {
    max-width: 78%;
    padding: .85rem 1.1rem;
    border-radius: 10px;
    font-family: 'Cairo', sans-serif;
    font-size: .9rem;
    line-height: 1.8;
    word-break: break-word;
    position: relative;
}
.bubble-user {
    align-self: flex-end;
    background: rgba(0,68,255,.15);
    border: 1px solid rgba(0,68,255,.3);
    color: var(--text);
    direction: rtl;
    text-align: right;
    border-bottom-right-radius: 3px;
}
.bubble-assistant {
    align-self: flex-start;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    direction: rtl;
    text-align: right;
    border-bottom-left-radius: 3px;
}
.bubble-assistant::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 10px 10px 0 0;
    background: linear-gradient(90deg, var(--success), var(--accent));
}
.bubble-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .55rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .35rem;
    direction: ltr;
    text-align: left;
}

.answer-text {
    font-family: 'Cairo', 'IBM Plex Mono', sans-serif;
    font-size: .95rem;
    line-height: 1.95;
    color: var(--text);
    direction: rtl;
    text-align: right;
    unicode-bidi: plaintext;
    word-break: break-word;
}

.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: .9rem !important;
    padding: .8rem 1.1rem !important;
    direction: rtl !important;
    text-align: right !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: var(--glow-c) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--muted) !important;
    direction: rtl !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: .78rem !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    padding: .65rem 1.8rem !important;
    transition: opacity .15s, transform .1s, box-shadow .2s !important;
    box-shadow: 0 4px 20px rgba(0,68,255,.3) !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,68,255,.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.status-bar {
    display: flex;
    align-items: flex-start;
    gap: .85rem;
    padding: .85rem 1.1rem;
    border-radius: 6px;
    margin: .5rem 0 .85rem;
    font-size: .78rem;
}
.status-action {
    background: rgba(245,166,35,.07);
    border: 1px solid rgba(245,166,35,.22);
}
.status-ok {
    background: rgba(0,200,240,.05);
    border: 1px solid rgba(0,200,240,.15);
}
.status-msg {
    color: var(--muted2);
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
    flex: 1;
    line-height: 1.7;
}
.status-msg .problem-line {
    color: var(--warn);
    font-size: .8rem;
    margin-top: .3rem;
    display: block;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    font-size: .65rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: .09em;
    padding: .25rem .7rem;
    border-radius: 3px;
    white-space: nowrap;
}
.chip-action {
    background: rgba(245,166,35,.1);
    border: 1px solid rgba(245,166,35,.3);
    color: var(--warn);
}
.chip-ok {
    background: rgba(0,200,240,.08);
    border: 1px solid rgba(0,200,240,.22);
    color: var(--accent);
}

.source-item {
    font-size: .72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted2);
    padding: .4rem 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: .5rem;
}
.source-item:last-child { border-bottom: none; }

.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.stat-box {
    flex: 1;
    min-width: 130px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
.stat-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin-bottom: .25rem;
}
.stat-lbl {
    font-size: .62rem;
    color: var(--muted);
    letter-spacing: .12em;
    text-transform: uppercase;
}

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px !important; overflow: hidden !important; }
.stDataFrame table { font-family: 'IBM Plex Mono', monospace !important; font-size: .75rem !important; }

hr { border-color: var(--border) !important; margin: 2.5rem 0 !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }

h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    color: var(--text) !important;
    margin: 1.5rem 0 .5rem !important;
}
div[data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =========================
#  HERO
# =========================
st.markdown("""
<div class="hero-wrap">
  <div class="hero-inner">
    <div class="hero-text">
      <div class="hero-badge">📡 &nbsp; Telecom Intelligence Platform</div>
      <div class="hero-title">NileTel<br>Assistant</div>
      <div class="hero-sub">
        <span class="signal-dot"></span>
        <span>System online</span>
        <span class="sep">·</span>
        <span>AI-powered support</span>
        <span class="sep">·</span>
        <span>Real-time tickets</span>
      </div>
    </div>
    <svg class="hero-illus" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="100" cy="100" rx="90" ry="90" stroke="rgba(0,200,240,0.08)" stroke-width="1"/>
      <ellipse cx="100" cy="100" rx="65" ry="65" stroke="rgba(0,200,240,0.12)" stroke-width="1"/>
      <ellipse cx="100" cy="100" rx="40" ry="40" stroke="rgba(0,200,240,0.18)" stroke-width="1"/>
      <path d="M130 70 Q145 85 130 100" stroke="rgba(0,232,135,0.5)" stroke-width="1.5" stroke-linecap="round" fill="none"/>
      <path d="M138 62 Q160 81 138 108" stroke="rgba(0,232,135,0.3)" stroke-width="1.5" stroke-linecap="round" fill="none"/>
      <path d="M146 54 Q175 77 146 116" stroke="rgba(0,232,135,0.15)" stroke-width="1.5" stroke-linecap="round" fill="none"/>
      <rect x="78" y="82" width="44" height="36" rx="5" fill="#0f1e3a" stroke="rgba(0,200,240,0.6)" stroke-width="1.5"/>
      <rect x="84" y="88" width="32" height="20" rx="3" fill="rgba(0,200,240,0.06)" stroke="rgba(0,200,240,0.2)" stroke-width="1"/>
      <rect x="86" y="90" width="14" height="6" rx="1" fill="rgba(0,200,240,0.25)"/>
      <rect x="86" y="98" width="22" height="3" rx="1" fill="rgba(0,200,240,0.15)"/>
      <rect x="86" y="103" width="16" height="3" rx="1" fill="rgba(0,200,240,0.1)"/>
      <rect x="38" y="90" width="34" height="20" rx="3" fill="#0a1628" stroke="rgba(0,68,255,0.5)" stroke-width="1"/>
      <line x1="47" y1="90" x2="47" y2="110" stroke="rgba(0,68,255,0.3)" stroke-width="1"/>
      <line x1="55" y1="90" x2="55" y2="110" stroke="rgba(0,68,255,0.3)" stroke-width="1"/>
      <line x1="63" y1="90" x2="63" y2="110" stroke="rgba(0,68,255,0.3)" stroke-width="1"/>
      <rect x="72" y="97" width="6" height="6" rx="1" fill="rgba(0,68,255,0.3)"/>
      <rect x="128" y="90" width="34" height="20" rx="3" fill="#0a1628" stroke="rgba(0,68,255,0.5)" stroke-width="1"/>
      <line x1="137" y1="90" x2="137" y2="110" stroke="rgba(0,68,255,0.3)" stroke-width="1"/>
      <line x1="145" y1="90" x2="145" y2="110" stroke="rgba(0,68,255,0.3)" stroke-width="1"/>
      <line x1="153" y1="90" x2="153" y2="110" stroke="rgba(0,68,255,0.3)" stroke-width="1"/>
      <rect x="122" y="97" width="6" height="6" rx="1" fill="rgba(0,68,255,0.3)"/>
      <line x1="100" y1="82" x2="100" y2="60" stroke="rgba(0,200,240,0.7)" stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="100" cy="58" r="3" fill="#00c8f0" opacity="0.9"/>
      <circle cx="100" cy="58" r="6" fill="rgba(0,200,240,0.15)"/>
      <circle cx="30"  cy="30"  r="1"   fill="rgba(255,255,255,0.6)"/>
      <circle cx="170" cy="25"  r="1.5" fill="rgba(255,255,255,0.4)"/>
      <circle cx="20"  cy="155" r="1"   fill="rgba(255,255,255,0.5)"/>
      <circle cx="175" cy="160" r="1"   fill="rgba(255,255,255,0.6)"/>
      <circle cx="55"  cy="170" r="1"   fill="rgba(0,200,240,0.6)"/>
      <circle cx="155" cy="40"  r="1"   fill="rgba(0,232,135,0.6)"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
#  STATS ROW
# =========================
st.markdown("""
<div class="stats-row">
  <div class="stat-box">
    <div class="stat-num">99.8<span style="font-size:.9rem;color:var(--muted2)">%</span></div>
    <div class="stat-lbl">Uptime</div>
  </div>
  <div class="stat-box">
    <div class="stat-num" style="color:var(--success);">&lt;2s</div>
    <div class="stat-lbl">Avg Response</div>
  </div>
  <div class="stat-box">
    <div class="stat-num" style="color:var(--accent3);">24/7</div>
    <div class="stat-lbl">AI Support</div>
  </div>
  <div class="stat-box">
    <div class="stat-num" style="color:var(--warn);">Live</div>
    <div class="stat-lbl">Ticket Tracking</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
#  CHAT HISTORY DISPLAY
# =========================
if st.session_state.chat_history:
    st.markdown('<div class="section-label">// سجل المحادثة</div>', unsafe_allow_html=True)

    bubbles_html = '<div class="bubble-wrap">'
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            bubbles_html += (
                f'<div class="bubble bubble-user">'
                f'<div class="bubble-label">أنت</div>'
                f'{msg["text"]}'
                f'</div>'
            )
        else:
            bubbles_html += (
                f'<div class="bubble bubble-assistant">'
                f'<div class="bubble-label">NileTel Assistant</div>'
                f'<div class="answer-text" dir="rtl" lang="ar">{msg["text"]}</div>'
                f'</div>'
            )
    bubbles_html += '</div>'
    st.markdown(bubbles_html, unsafe_allow_html=True)

    # Show action status & sources only for the latest assistant reply
    last_assistant = next(
        (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
        None,
    )
    if last_assistant:
        if last_assistant.get("needs_action") == "YES":
            # ── FIX 2: show the recorded problem inside the ticket status bar ──
            ticket_problem = last_assistant.get("ticket_problem", "")
            problem_html   = (
                f'<span class="problem-line">📋 المشكلة: {ticket_problem}</span>'
                if ticket_problem else ""
            )
            st.markdown(
                f'<div class="status-bar status-action">'
                f'<span class="chip chip-action">⚡ تذكرة مُنشأة</span>'
                f'<span class="status-msg">'
                f'تم رفع تذكرة دعم وسيتم التواصل معك قريباً.'
                f'{problem_html}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-bar status-ok">'
                '<span class="chip chip-ok">✓ No Action Needed</span>'
                '<span class="status-msg">تم الرد على استفسارك بدون الحاجة لتصعيد.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        sources = last_assistant.get("sources", [])
        if sources:
            src_list = sources if isinstance(sources, list) else [sources]
            src_html = "".join(
                f'<div class="source-item">'
                f'<span style="color:var(--accent);font-size:.7rem">▸</span>{s}'
                f'</div>'
                for s in src_list
            )
            st.markdown(
                '<div class="section-label" style="margin-top:1rem">// مصادر المعلومات</div>'
                f'<div class="card" style="padding:1rem 1.75rem">{src_html}</div>',
                unsafe_allow_html=True,
            )

# =========================
#  QUERY INPUT
# =========================
st.markdown(
    '<div class="section-label" style="margin-top:1.5rem">// واجهة الاستفسار</div>',
    unsafe_allow_html=True,
)

query = st.text_input(
    label="",
    placeholder="مثال: ليه النت بطيء؟ / إزاي أجدد الباقة؟",
    label_visibility="collapsed",
    key="query_input",
)

col_btn, col_clear, col_spacer = st.columns([1, 1, 5])
with col_btn:
    send = st.button("إرسال  ⟶", use_container_width=True)
with col_clear:
    clear = st.button("🗑 مسح المحادثة", use_container_width=True)

# ── Clear history ──
if clear:
    st.session_state.chat_history = []
    st.session_state.session_id   = str(uuid.uuid4())  # fresh session = fresh backend memory
    st.rerun()

# ── Handle Send ──
if send:
    if not query.strip():
        st.warning("من فضلك اكتب سؤالك قبل الإرسال.")
    else:
        # Optimistically add user message
        st.session_state.chat_history.append({"role": "user", "text": query})

        with st.spinner("جاري الاتصال بنظام الذكاء الاصطناعي..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query":      query,
                        "session_id": st.session_state.session_id,
                    },
                    headers={"ngrok-skip-browser-warning": "true"},
                    timeout=15,
                )

                if response.status_code == 200:
                    data = response.json()

                    answer_text    = data.get("answer",         "لا توجد إجابة.")
                    needs_action   = data.get("needs_action",   "NO")
                    sources        = data.get("sources",        [])
                    ticket_problem = data.get("ticket_problem", "")  

                    st.session_state.chat_history.append({
                        "role":           "assistant",
                        "text":           answer_text,
                        "needs_action":   needs_action,
                        "sources":        sources,
                        "ticket_problem": ticket_problem,          
                    })
                    st.rerun()

                else:
                    st.session_state.chat_history.pop()
                    st.error(f"خطأ في الـ API: {response.status_code} — {response.text}")

            except requests.exceptions.ConnectionError:
                st.session_state.chat_history.pop()
                st.error("تعذّر الاتصال بالـ API. تأكد أن الخادم يعمل على localhost:8000")
            except requests.exceptions.Timeout:
                st.session_state.chat_history.pop()
                st.error("انتهت مهلة الاتصال. حاول مرة أخرى.")
            except Exception as e:
                st.session_state.chat_history.pop()
                st.error(f"خطأ غير متوقع: {e}")

# =========================
# 🎫 TICKET MANAGEMENT
# =========================
st.markdown("---")
st.markdown('<div class="section-label">// إدارة التذاكر</div>', unsafe_allow_html=True)

col_load, col_info = st.columns([1, 4])
with col_load:
    load_btn = st.button("↓  تحميل التذاكر", use_container_width=True)
with col_info:
    st.markdown(
        '<div style="color:var(--muted2);font-size:.75rem;padding-top:.6rem;'
        'font-family:Cairo,sans-serif;direction:rtl;text-align:right;">'
        'بيانات مباشرة من جدول عمليات NileTel</div>',
        unsafe_allow_html=True,
    )

if load_btn:
    with st.spinner("جاري تحميل بيانات التذاكر..."):
        try:
            df = pd.read_csv(SHEET_URL)
            st.markdown(
                f'<span class="chip chip-ok">✓ &nbsp;{len(df)} تذكرة محملة</span>',
                unsafe_allow_html=True,
            )
            st.dataframe(df, use_container_width=True, hide_index=False)
        except Exception as e:
            st.error(f"فشل تحميل التذاكر: {e}")