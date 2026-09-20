import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Realty Lens", page_icon="", layout="wide")
st.title("Relens - The brochure based Assitant")
st.caption("Ask questions about uploaded brochures. Answers are grounded strictly in the uploaded documents.")


def fetch_documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.sidebar.error(f"Could not reach backend: {exc}")
        return []


def render_table(table_data):
    df = pd.DataFrame(table_data).drop(columns=["reason"], errors="ignore")
    df.index = range(1, len(df) + 1)
    st.dataframe(df, use_container_width=True)


def render_sources(sources):
    if not sources:
        return
    grouped = {}
    for s in sources:
        key = (s.get("project_name") or s["source_file"], s["source_file"])
        grouped.setdefault(key, set())
        if s.get("page"):
            grouped[key].add(s["page"])

    with st.expander("Sources"):
        for (label, source_file), pages in grouped.items():
            if pages:
                page_str = ", ".join(str(p) for p in sorted(pages))
                suffix = f", pp. {page_str}" if len(pages) > 1 else f", p. {page_str}"
            else:
                suffix = ""
            st.write(f"- {label} ({source_file}{suffix})")


with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload brochure PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("Ingest"):
        with st.spinner("Ingesting (text extraction, vision fallback, embedding, structured extraction)..."):
            files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
            try:
                resp = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=600)
                resp.raise_for_status()
                for doc in resp.json()["documents"]:
                    label = doc.get("project_name") or doc["source_file"]
                    scanned = " (scanned — vision used)" if doc.get("is_scanned") else ""
                    st.success(f"Ingested: {label}{scanned}")
                    for warning in doc.get("warnings", []):
                        st.warning(warning)
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    docs = fetch_documents()
    if docs:
        st.write(f"**{len(docs)} project(s) ingested:**")
        for d in docs:
            label = d.get("project_name") or d["source_file"]
            tag = " 📷" if d.get("is_scanned") else ""
            st.write(f"- {label}{tag}")
    else:
        st.write("No documents ingested yet.")

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.get("history_loaded"):
    # Only mark as loaded on a successful fetch — if the backend happens to be mid-restart
    # right now, retry on the next rerun instead of permanently starting the session empty.
    try:
        resp = requests.get(f"{BACKEND_URL}/history", timeout=10)
        resp.raise_for_status()
        for entry in resp.json():
            st.session_state.history.append({"role": "user", "content": entry["question"]})
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": entry["answer_text"],
                    "caption": f"intent: {entry['intent']}",
                    "table": entry.get("table_data"),
                    "sources": entry.get("sources"),
                }
            )
        st.session_state.history_loaded = True
    except requests.RequestException:
        pass

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("caption"):
            st.caption(turn["caption"])
        if turn.get("table"):
            render_table(turn["table"])
        render_sources(turn.get("sources"))

question = st.chat_input("Ask about the uploaded brochures...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/query", json={"question": question}, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                st.write(data["answer_text"])
                st.caption(f"intent: {data['intent']}")
                if data.get("table_data"):
                    render_table(data["table_data"])
                render_sources(data.get("sources"))
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": data["answer_text"],
                        "caption": f"intent: {data['intent']}",
                        "table": data.get("table_data"),
                        "sources": data.get("sources"),
                    }
                )
            except requests.RequestException as exc:
                error_msg = f"Request failed: {exc}"
                st.error(error_msg)
                st.session_state.history.append({"role": "assistant", "content": error_msg})
