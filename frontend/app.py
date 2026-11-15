"""
Streamlit page for MediAssist AI.

Run (backend must be running first):
    uvicorn backend.main:app --port 8000
    streamlit run frontend/app.py
"""

import sys
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


# ======================= login / sign up =======================
if not st.session_state.token:
    st.title("MediAssist AI")
    st.caption("Hospital information assistant. Please log in.")
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
                                            user={"username": result["username"], "role": result["role"]})
                    st.rerun()
    with signup_tab:
        try:
            role_options = api.roles()
        except api.ApiError:
            role_options = ["doctor", "front_desk", "insurance", "clinical_ops"]
        with st.form("signup"):
            new_name = st.text_input("Choose a username", help="3 to 30 characters: letters, numbers, dot, underscore")
            new_pass = st.text_input("Choose a password", type="password", help="At least 8 characters, letters and numbers")
            new_role = st.selectbox("Your role", role_options)
            if st.form_submit_button("Create account"):
                if call(api.signup, new_name, new_pass, new_role):
                    st.success("Account created. You can log in now.")
    st.stop()

token = st.session_state.token
user = st.session_state.user
is_admin = user["role"] == "admin"

# ======================= sidebar: files =======================
with st.sidebar:
    st.subheader(f"{user['username']}")
    st.caption(f"Role: {user['role']}")
    if st.button("Log out"):
        logout()
    st.divider()

    st.markdown("**Hospital documents (knowledge base)**")
    if is_admin:
        doc = st.file_uploader("Upload a PDF, DOCX or TXT", type=["pdf", "docx", "txt"],
                               key=f"doc_{st.session_state.upload_key}")
        if doc and st.button("Add to knowledge base"):
            with st.spinner("Reading and indexing the document..."):
                info = call(api.upload_document, token, doc.name, doc.getvalue(), mime_of(doc.name))
            if info:
                st.success(f"Indexed {info['chunks']} chunks from {info['source']}.")
                st.session_state.upload_key += 1
                st.rerun()
    else:
        st.caption("Only an admin can add or remove documents.")
    for source in call(api.sources, token) or []:
        left, right = st.columns([4, 1])
        left.write(f"{source['name']}  \n:gray[{source['chunks']} chunks]")
        if is_admin and right.button("✕", key=f"del_{source['name']}", help="Delete this file"):
            if call(api.delete_source, token, source["name"]):
                st.rerun()

    st.divider()
    st.markdown("**Medical images (prescriptions, lab reports)**")
    img = st.file_uploader("Upload a PNG or JPG", type=["png", "jpg", "jpeg"],
                           key=f"img_{st.session_state.upload_key}")
    if img and st.button("Analyze image"):
        with st.spinner("Reading the image (this can take 10 to 20 seconds)..."):
            result = call(api.upload_image, token, img.name, img.getvalue(), mime_of(img.name))
        if result:
            st.success("Image analyzed. Select it below to ask about it.")
            for warning in result.get("warnings", []):
                st.warning(warning)
            st.session_state.upload_key += 1
            st.rerun()
    image_list = call(api.images, token) or []
    for item in image_list:
        left, right = st.columns([4, 1])
        left.write(f"{item['name']}  \n:gray[{item['image_type'].replace('_', ' ')}]")
        if right.button("✕", key=f"delimg_{item['id']}", help="Delete this image result"):
            if call(api.delete_image, token, item["id"]):
                st.rerun()

# ======================= main: chat + dashboard =======================
st.title("MediAssist AI")
prompt = st.chat_input("Ask about hospital processes, or a patient (for example: lab results for P00001)")
chat_tab, eval_tab = st.tabs(["Chat", "Evaluation dashboard"])

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
        picked = turns[labels.index(st.selectbox("Question", labels, index=len(labels) - 1))]
        evaluation = picked["evaluation"] or {}

        st.subheader("Evaluation (LLM-as-a-Judge)")
        columns = st.columns(5)
        columns[0].metric("Confidence", helpers.show(evaluation.get("confidence")))
        columns[1].metric("Faithfulness (1-5)", helpers.show(evaluation.get("faithfulness")))
        columns[2].metric("Grounding (1-5)", helpers.show(evaluation.get("grounding")))
        columns[3].metric("Relevance (1-5)", helpers.show(evaluation.get("relevance")))
        columns[4].metric("Retrieval quality", helpers.show(evaluation.get("retrieval_quality")))
        st.write(f"**Decision:** {helpers.decision_text(evaluation)}")
        if evaluation.get("hallucination") is not None:
            st.write("**Hallucination detected:** " + ("yes" if evaluation["hallucination"] else "no"))
        if evaluation.get("judge_reason"):
            st.write(f"**Judge's reason:** {evaluation['judge_reason']}")
        if evaluation.get("guard_issues"):
            st.write("**Safety rule notes:** " + ", ".join(evaluation["guard_issues"]))

        st.subheader("Agent execution flow")
        rows = helpers.trace_rows(picked["trace"])
        if rows:
            frame = pd.DataFrame(rows)
            st.dataframe(frame, hide_index=True)
            st.bar_chart(frame, x="Agent", y="Seconds")
            st.caption(f"Total time: {picked['seconds']} s. The retriever, mcp and vision agents run at the same time.")

        st.subheader("Plan and retrieval trace")
        plan = picked["plan"] or {}
        if plan:
            needed = [label for label, on in (("hospital documents", plan.get("use_documents")),
                                              ("patient database", plan.get("use_database")),
                                              ("uploaded image", plan.get("use_image"))) if on]
            st.write("**Planner chose:** " + (", ".join(needed) or "nothing needed"))
            for call_info in plan.get("calls", []):
                st.write(f"• Database tool `{call_info['name']}` with {call_info['arguments']}")
            for tool in plan.get("missing", []):
                st.write(f"• `{tool}` was skipped: no patient ID")
        for line in helpers.source_lines(picked["sources"]):
            st.write("• Evidence used: " + line)

        st.subheader("This session")
        st.dataframe(pd.DataFrame([{
            "Question": t["question"][:70], "Decision": helpers.decision_text(t["evaluation"] or {}),
            "Confidence": helpers.show((t["evaluation"] or {}).get("confidence")), "Seconds": t["seconds"]}
            for t in turns]), hide_index=True)
