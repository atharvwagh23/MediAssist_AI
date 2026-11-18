"""
Streamlit page for MediAssist AI.

Run (backend must be running first):
    uvicorn backend.main:app --port 8000
    streamlit run frontend/app.py
"""

import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import app" works from this folder

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from frontend import api, helpers  # noqa: E402

MIME = {"pdf": "application/pdf", "txt": "text/plain", "png": "image/png", "jpg": "image/jpeg",
        "jpeg": "image/jpeg", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

st.set_page_config(page_title="MediAssist AI", page_icon="🩺", layout="wide")

for key, default in (("token", None), ("user", None), ("turns", []), ("upload_key", 0), ("notice", None)):
    st.session_state.setdefault(key, default)

# colour of each status (r, g, b)
TONES = {"good": (16, 185, 129), "warn": (245, 158, 11), "bad": (239, 68, 68), "none": (148, 163, 184)}
AGENT_COLOURS = ["#14b8a6", "#3b82f6", "#8b5cf6", "#ec4899", "#f97316", "#22c55e"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700;800&display=swap');

/* ================= page background + fonts ================= */
.stApp {
    background:
        radial-gradient(900px 500px at 8% 0%, rgba(45,212,191,.22), transparent 60%),
        radial-gradient(800px 500px at 95% 8%, rgba(99,102,241,.16), transparent 60%),
        linear-gradient(160deg, #ecfeff 0%, #f0f9ff 50%, #eef2ff 100%);
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stBottom"] > div { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }

.stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button,
.stApp [data-testid="stCaptionContainer"], .stApp .stMarkdown { font-family: 'Inter', sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Poppins', sans-serif; letter-spacing: -.01em; }

/* gradient title */
.stApp h1 {
    font-weight: 800; font-size: 2.8rem;
    background: linear-gradient(90deg, #0f766e 0%, #0ea5e9 55%, #6366f1 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.tagline { margin: -10px 0 14px; font-size: 1rem; color: #475569; font-family: 'Inter', sans-serif; }

/* ================= tabs as pills ================= */
div[data-baseweb="tab-list"] { gap: 10px; border-bottom: none; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }
button[data-baseweb="tab"] {
    background: rgba(255,255,255,.75); border: 1px solid rgba(15,118,110,.25);
    border-radius: 999px; padding: 8px 22px; height: auto;
}
button[data-baseweb="tab"] p { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: .95rem; }
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #0f766e, #0ea5e9); border-color: transparent;
    box-shadow: 0 6px 16px rgba(14,165,233,.3);
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #fff; }

/* ================= chat ================= */
[data-testid="stChatMessage"] {
    background: #ffffff; border: 1px solid rgba(15,118,110,.15); border-radius: 18px;
    padding: 16px 20px; box-shadow: 0 6px 18px rgba(15,23,42,.06); margin-bottom: 12px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #ccfbf1, #e0f2fe); border-color: rgba(14,165,233,.3);
}
[data-testid="stChatInput"] {
    border-radius: 18px; background: #ffffff; border: 1px solid rgba(15,118,110,.35);
    box-shadow: 0 8px 24px rgba(15,23,42,.10);
}
div[data-baseweb="select"] > div { background: #ffffff; border-radius: 12px; border-color: rgba(15,118,110,.3); }
.stApp [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #0f766e, #0ea5e9); color: #fff; border: none; border-radius: 10px;
    font-weight: 600;
}
.stApp [data-testid="stFormSubmitButton"] button p { color: #fff; }

/* ================= sidebar (dark teal) ================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2f3a 0%, #0b4a57 50%, #0f766e 100%);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #e6fffb; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.2); }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.08); border: 1px dashed rgba(255,255,255,.45); border-radius: 14px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * { color: #a7f3d0; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] { color: #ffffff; }
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.4); border-radius: 10px;
}
section[data-testid="stSidebar"] .stButton button *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { color: #ffffff; }
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,.28); }
section[data-testid="stSidebar"] [data-testid="stAlert"] * { color: #0f172a; }
.side-h { font-family: 'Poppins', sans-serif; font-size: .85rem; text-transform: uppercase;
          letter-spacing: .12em; font-weight: 700; color: #5eead4; margin: 2px 0 0; }
.fname { color: #ffffff; font-weight: 600; font-size: .9rem; word-break: break-word; }
.fmeta { color: #99f6e4; font-size: .75rem; margin-top: 1px; }

/* ================= dashboard ================= */
.hero-band { background: linear-gradient(120deg, #0f766e 0%, #0ea5e9 55%, #6366f1 100%);
             border-radius: 20px; padding: 24px 28px; color: white; margin-bottom: 18px;
             box-shadow: 0 10px 28px rgba(14,165,233,.30); }
.hero-band .t { font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; opacity: .85; }
.hero-band .q { font-family: 'Poppins', sans-serif; font-size: 1.3rem; font-weight: 600; margin-top: 6px; line-height: 1.4; }

.result { display: flex; gap: 24px; align-items: center; border-radius: 20px; padding: 22px 26px;
          background-color: #ffffff; box-shadow: 0 6px 18px rgba(15,23,42,.06); }
.ring { position: relative; width: 112px; height: 112px; flex: none; }
.ring svg { transform: rotate(-90deg); }
.ring .c { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring .c b { font-family: 'Poppins', sans-serif; font-size: 1.6rem; line-height: 1; }
.ring .c span { font-size: .62rem; opacity: .6; text-transform: uppercase; letter-spacing: .08em; margin-top: 3px; }
.result .ttl { font-family: 'Poppins', sans-serif; font-size: 1.2rem; font-weight: 700; color: #0f172a; }
.result .txt { font-size: .9rem; color: #334155; margin-top: 6px; line-height: 1.55; }
.pill { display: inline-block; font-size: .72rem; font-weight: 700; padding: 3px 12px; border-radius: 999px;
        margin: 0 6px 8px 0; color: white; }

.score { border-radius: 18px; padding: 18px 20px 16px; height: 100%; background-color: #ffffff;
         box-shadow: 0 6px 18px rgba(15,23,42,.06); }
.score .top { display: flex; justify-content: space-between; align-items: center; }
.score .lbl { font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #475569; }
.score .ico { font-size: 1.25rem; }
.score .val { font-family: 'Poppins', sans-serif; font-size: 2.2rem; font-weight: 800; line-height: 1.15; margin-top: 6px; }
.score .val small { font-size: .95rem; font-weight: 500; opacity: .6; }
.score .track { height: 8px; border-radius: 4px; background: rgba(100,116,139,.2); margin-top: 10px; overflow: hidden; }
.score .fill { height: 100%; border-radius: 4px; }
.score .sub { font-size: .75rem; color: #64748b; margin-top: 8px; }

.sec { display: flex; align-items: center; gap: .6rem; margin: 2.2rem 0 .9rem; }
.sec .ic { font-size: 1.25rem; }
.sec h3 { margin: 0; font-size: 1.2rem; font-weight: 700; color: #0f172a; }
.sec .hint { margin-left: auto; font-size: .78rem; color: #64748b; }

.flow { border-radius: 18px; padding: 8px 20px; background: #ffffff; border: 1px solid rgba(15,118,110,.15);
        box-shadow: 0 6px 18px rgba(15,23,42,.06); }
.step { display: grid; grid-template-columns: 32px 120px 1fr 70px; gap: 14px; align-items: center;
        padding: 13px 0; border-bottom: 1px dashed rgba(100,116,139,.3); }
.step:last-child { border-bottom: none; }
.step .n { width: 30px; height: 30px; border-radius: 50%; color: white; font-size: .8rem; font-weight: 700;
           display: flex; align-items: center; justify-content: center; }
.step .ag { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: .92rem; text-transform: capitalize; color: #0f172a; }
.step .tl { height: 10px; border-radius: 5px; background: rgba(100,116,139,.18); overflow: hidden; }
.step .tl > div { height: 100%; border-radius: 5px; }
.step .note { font-size: .78rem; color: #64748b; margin-top: 5px; }
.step .tm { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; font-size: .9rem; }

.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.chip { font-size: .82rem; font-weight: 600; padding: 6px 14px; border-radius: 999px; color: #0f172a;
        background: rgba(20,184,166,.18); border: 1px solid rgba(20,184,166,.6); }
.chip.off { background: transparent; border: 1px dashed rgba(100,116,139,.5); opacity: .5; font-weight: 500; }
.chip.skip { background: rgba(245,158,11,.18); border-color: rgba(245,158,11,.6); }
.chip.tool { background: rgba(99,102,241,.18); border-color: rgba(99,102,241,.6); }

@media (max-width: 700px) { .step { grid-template-columns: 32px 1fr 60px; } .step .tl { display: none; } }
</style>
""", unsafe_allow_html=True)


# ======================= small helpers =======================
def mime_of(name: str) -> str:
    return MIME.get(name.rsplit(".", 1)[-1].lower(), "application/octet-stream")


def logout() -> None:
    for key in ("token", "user", "turns", "notice"):
        st.session_state.pop(key, None)
    st.rerun()


def call(function, *args, **kwargs):
    """Run an API call. A 401 logs the user out; any other error is shown and returns None."""
    try:
        return function(*args, **kwargs)
    except api.ApiError as error:
        if error.status == 401 and st.session_state.token:
            st.session_state.notice = "Your session ended. Please log in again."
            logout()
        st.error(error.message)
        return None


def rgba(name: str, alpha: float) -> str:
    r, g, b = TONES[name]
    return f"rgba({r},{g},{b},{alpha})"


def tone(value, good: float, warn: float) -> str:
    """green at or above `good`, amber at or above `warn`, otherwise red, grey when missing."""
    if value is None:
        return "none"
    return "good" if value >= good else "warn" if value >= warn else "bad"


def section(icon: str, title: str, hint: str = "") -> None:
    st.markdown(f'<div class="sec"><span class="ic">{icon}</span><h3>{escape(title)}</h3>'
                f'<span class="hint">{escape(hint)}</span></div>', unsafe_allow_html=True)


def score_card(icon: str, label: str, shown: str, status: str, percent: float | None = None,
               top: float | None = None, sub: str = "") -> str:
    solid = rgba(status, 1)
    suffix = f"<small> / {top:g}</small>" if top else ""
    track = ""
    if percent is not None:
        track = (f'<div class="track"><div class="fill" '
                 f'style="width:{max(0, min(100, percent)):.0f}%;background:{solid}"></div></div>')
    return (f'<div class="score" style="background-image:linear-gradient(135deg,{rgba(status, .20)},{rgba(status, .04)});'
            f'border:1px solid {rgba(status, .55)}">'
            f'<div class="top"><span class="lbl">{escape(label)}</span><span class="ico">{icon}</span></div>'
            f'<div class="val" style="color:{solid}">{escape(shown)}{suffix}</div>{track}'
            f'<div class="sub">{escape(sub)}</div></div>')


def decision_tone(evaluation: dict) -> str:
    decision = evaluation.get("decision", "")
    if decision == "allowed":
        return "good"
    if decision in ("allowed_rules_only", "no_answer"):
        return "warn"
    return "bad" if decision else "none"


def ring(confidence, status: str) -> str:
    circumference = 2 * 3.14159 * 46
    filled = circumference * (confidence or 0)
    shown = "n/a" if confidence is None else f"{confidence * 100:.0f}%"
    return (f'<div class="ring"><svg width="112" height="112" viewBox="0 0 112 112">'
            f'<circle cx="56" cy="56" r="46" fill="none" stroke="rgba(100,116,139,.2)" stroke-width="10"/>'
            f'<circle cx="56" cy="56" r="46" fill="none" stroke="{rgba(status, 1)}" stroke-width="10" '
            f'stroke-linecap="round" stroke-dasharray="{filled:.1f} {circumference:.1f}"/></svg>'
            f'<div class="c"><b style="color:{rgba(status, 1)}">{shown}</b><span>confidence</span></div></div>')


def file_label(name: str, meta: str) -> str:
    return f'<div class="fname">{escape(name)}</div><div class="fmeta">{escape(meta)}</div>'


# ======================= login / sign up =======================
if not st.session_state.token:
    st.title("MediAssist AI")
    st.markdown('<div class="tagline">Hospital information assistant. Please log in.</div>', unsafe_allow_html=True)
    if st.session_state.notice:
        st.warning(st.session_state.notice)
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
    with login_tab:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                result = call(api.login, username, password)
                if result:
                    st.session_state.update(token=result["token"], notice=None,
                                            user={"username": result["username"]})
                    st.rerun()
    with signup_tab:
        with st.form("signup"):
            new_name = st.text_input("Choose a username", help="3 to 30 characters: letters, numbers, dot, underscore")
            new_pass = st.text_input("Choose a password", type="password", help="At least 8 characters, letters and numbers")
            if st.form_submit_button("Create account"):
                if call(api.signup, new_name, new_pass):
                    st.success("Account created. You can log in now.")
    st.stop()

token = st.session_state.token
user = st.session_state.user

# ======================= sidebar: medical images first, then documents =======================
with st.sidebar:
    st.subheader(f"👤 {user['username']}")
    if st.button("Log out"):
        logout()
    st.divider()

    # ---- Medical images (top) ----
    st.markdown('<div class="side-h">🖼️ Medical images</div>', unsafe_allow_html=True)
    st.caption("Prescriptions and lab reports")
    img = st.file_uploader("Upload a PNG or JPG", type=["png", "jpg", "jpeg"],
                           key=f"img_{st.session_state.upload_key}")
    if img and st.button("Analyze image", use_container_width=True):
        with st.spinner("Reading the image (this can take 10 to 20 seconds)..."):
            result = call(api.upload_image, token, img.name, img.getvalue(), mime_of(img.name))
        if result:
            st.success("Image analyzed. Select it in the chat tab to ask about it.")
            for warning in result.get("warnings", []):
                st.warning(warning)
            st.session_state.upload_key += 1
            st.rerun()
    image_list = call(api.images, token) or []
    for item in image_list:
        left, right = st.columns([4, 1])
        left.markdown(file_label(item["name"], item["image_type"].replace("_", " ")), unsafe_allow_html=True)
        if right.button("✕", key=f"delimg_{item['id']}", help="Delete this image result"):
            if call(api.delete_image, token, item["id"]):
                st.rerun()

    st.divider()

    # ---- Hospital documents ----
    st.markdown('<div class="side-h">📄 Hospital documents</div>', unsafe_allow_html=True)
    st.caption("Knowledge base")
    doc = st.file_uploader("Upload a PDF, DOCX or TXT", type=["pdf", "docx", "txt"],
                           key=f"doc_{st.session_state.upload_key}")
    if doc and st.button("Add to knowledge base", use_container_width=True):
        with st.spinner("Reading and indexing the document..."):
            info = call(api.upload_document, token, doc.name, doc.getvalue(), mime_of(doc.name))
        if info:
            st.success(f"Indexed {info['chunks']} chunks from {info['source']}.")
            st.session_state.upload_key += 1
            st.rerun()
    for source in call(api.sources, token) or []:
        left, right = st.columns([4, 1])
        left.markdown(file_label(source["name"], f"{source['chunks']} chunks"), unsafe_allow_html=True)
        if right.button("✕", key=f"del_{source['name']}", help="Delete this file"):
            if call(api.delete_source, token, source["name"]):
                st.rerun()

# ======================= main: chat + dashboard =======================
st.title("MediAssist AI")
st.markdown('<div class="tagline">Ask about hospital processes, patient records and uploaded reports.</div>',
            unsafe_allow_html=True)
prompt = st.chat_input("Ask about hospital processes, or a patient (for example: lab results for P00001)")
chat_tab, eval_tab = st.tabs(["💬 Chat", "📊 Evaluation dashboard"])

with chat_tab:
    choices = {"No image": None} | {f"{i['name']} ({i['image_type'].replace('_', ' ')})": i["id"] for i in image_list}
    chosen = st.selectbox("Use an uploaded image in your next question", list(choices))
    st.caption(f"Showing the last {config.MAX_HISTORY} questions and answers (the oldest is dropped first).")

    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn["sources"]:
                with st.expander("Sources"):
                    for line in helpers.source_lines(turn["sources"]):
                        st.write("• " + line)
            st.caption(f"Confidence: {helpers.show(turn['confidence'])}  |  {turn['seconds']} s")

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("The agents are working..."):
                data = call(api.chat, token, prompt, choices[chosen],
                            helpers.history_for_backend(st.session_state.turns))
            if data:
                st.write_stream(helpers.typewriter(data["answer"]))
                if data["sources"]:
                    with st.expander("Sources"):
                        for line in helpers.source_lines(data["sources"]):
                            st.write("• " + line)
                st.caption(f"Confidence: {helpers.show(data.get('confidence'))}  |  {data.get('seconds')} s")
                st.session_state.turns = helpers.push_turn(st.session_state.turns, {
                    "question": prompt, "answer": data["answer"], "sources": data["sources"],
                    "confidence": data.get("confidence"), "evaluation": data.get("evaluation", {}),
                    "trace": data.get("trace", []), "plan": data.get("plan", {}),
                    "seconds": data.get("seconds"), "guard": data.get("guard")}, config.MAX_HISTORY)

with eval_tab:
    turns = st.session_state.turns
    if not turns:
        st.info("Ask a question first. Its agent flow and evaluation scores will appear here.")
    else:
        labels = [f"{i + 1}. {t['question'][:60]}" for i, t in enumerate(turns)]
        picked = turns[labels.index(st.selectbox("Choose a question", labels, index=len(labels) - 1))]
        evaluation = picked["evaluation"] or {}
        confidence = evaluation.get("confidence")
        status = decision_tone(evaluation)
        hallucination = evaluation.get("hallucination")
        seconds = picked.get("seconds")

        # ---------- header band ----------
        st.markdown(f'<div class="hero-band"><div class="t">Evaluating this question</div>'
                    f'<div class="q">{escape(picked["question"])}</div></div>', unsafe_allow_html=True)

        # ---------- headline result: ring + decision + reasons ----------
        pills = ""
        if hallucination is not None:
            pills += (f'<span class="pill" style="background:{rgba("bad" if hallucination else "good", 1)}">'
                      f'{"Hallucination detected" if hallucination else "No hallucination"}</span>')
        pills += f'<span class="pill" style="background:#6366f1">{len(picked["sources"] or [])} sources</span>'
        if seconds is not None:
            pills += f'<span class="pill" style="background:#0ea5e9">{seconds} s</span>'
        details = []
        if evaluation.get("judge_reason"):
            details.append("<b>Judge:</b> " + escape(evaluation["judge_reason"]))
        if evaluation.get("guard_issues"):
            details.append("<b>Safety notes:</b> " + escape(", ".join(evaluation["guard_issues"])))
        st.markdown(
            f'<div class="result" style="background-image:linear-gradient(135deg,{rgba(status, .16)},{rgba(status, .03)});'
            f'border:1px solid {rgba(status, .5)}">{ring(confidence, status)}<div>{pills}'
            f'<div class="ttl">{escape(helpers.decision_text(evaluation))}</div>'
            f'<div class="txt">{"<br>".join(details) or "No further notes."}</div></div></div>',
            unsafe_allow_html=True)

        # ---------- score cards ----------
        section("🎯", "Answer quality", "scored by the LLM judge, 1 (poor) to 5 (excellent)")
        cards = st.columns(4)
        for column, (icon, label, key, sub) in zip(cards, (
                ("🧾", "Faithfulness", "faithfulness", "Claims backed by evidence"),
                ("📌", "Grounding", "grounding", "Correct source labels"),
                ("💡", "Relevance", "relevance", "Answers the question"))):
            value = evaluation.get(key)
            column.markdown(score_card(icon, label, helpers.show(value), tone(value, 4, 3),
                                       None if value is None else value / 5 * 100, 5, sub),
                            unsafe_allow_html=True)
        quality = evaluation.get("retrieval_quality")
        cards[3].markdown(score_card("🔍", "Retrieval", helpers.show(quality), tone(quality, .7, .5),
                                     None if quality is None else quality * 100, 1, "Quality of what was found"),
                          unsafe_allow_html=True)

        # ---------- agent execution flow ----------
        section("🤖", "Agent execution flow", "retriever, mcp and vision run at the same time")
        rows = helpers.trace_rows(picked["trace"])
        if rows:
            longest = max((r["Seconds"] or 0) for r in rows) or 1
            steps = ""
            for n, r in enumerate(rows, 1):
                colour = AGENT_COLOURS[(n - 1) % len(AGENT_COLOURS)]
                width = max(4, (r["Seconds"] or 0) / longest * 100)
                steps += (f'<div class="step"><div class="n" style="background:{colour}">{n}</div>'
                          f'<div class="ag">{escape(str(r["Agent"]))}</div>'
                          f'<div><div class="tl"><div style="width:{width:.0f}%;background:{colour}"></div></div>'
                          f'<div class="note">{escape(str(r["What it did"]))}</div></div>'
                          f'<div class="tm" style="color:{colour}">{r["Seconds"]} s</div></div>')
            st.markdown(f'<div class="flow">{steps}</div>', unsafe_allow_html=True)
        else:
            st.caption("No agent trace for this answer.")

        # ---------- plan + retrieved evidence ----------
        section("🧭", "Plan and retrieved evidence")
        plan = picked["plan"] or {}
        if plan:
            chips = "".join(
                f'<span class="chip{"" if on else " off"}">{label}</span>'
                for label, on in (("📄 Hospital documents", plan.get("use_documents")),
                                  ("🗄️ Patient database", plan.get("use_database")),
                                  ("🖼️ Uploaded image", plan.get("use_image"))))
            for call_info in plan.get("calls", []):
                chips += f'<span class="chip tool">🔧 {escape(str(call_info["name"]))}</span>'
            for tool in plan.get("missing", []):
                chips += f'<span class="chip skip">⚠ {escape(str(tool))} skipped: no patient ID</span>'
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

        evidence = []
        for s in picked["sources"] or []:
            kind = s.get("type", "document")
            evidence.append({
                "Type": {"database": "🗄️ Database", "image": "🖼️ Image"}.get(kind, "📄 Document"),
                "Source": s.get("source"),
                "Detail": (f"patient {s['detail']}" if s.get("detail") else
                           (f"page {s.get('page')}" if kind == "document" else "")),
                "Score": s.get("score"),
            })
        if evidence:
            st.dataframe(pd.DataFrame(evidence), hide_index=True, use_container_width=True,
                         column_config={"Score": st.column_config.NumberColumn("Score", format="%.3f")})
        else:
            st.caption("No evidence was retrieved for this answer.")

        # ---------- session history ----------
        section("🗂️", "This session", f"last {config.MAX_HISTORY} questions")
        st.dataframe(pd.DataFrame([{
            "Question": t["question"][:70],
            "Decision": helpers.decision_text(t["evaluation"] or {}),
            "Confidence": (t["evaluation"] or {}).get("confidence"),
            "Seconds": t["seconds"]} for t in turns]),
            hide_index=True, use_container_width=True,
            column_config={"Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f")})