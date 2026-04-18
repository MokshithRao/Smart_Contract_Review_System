import streamlit as st
import requests


def render_streamlit_ui(api_url: str = "http://localhost:8000/analyze") -> None:
    st.set_page_config(page_title="Smart Contract Review", page_icon="🧠", layout="wide")
    st.title("Smart Contract Review System")
    st.write("Upload a contract file or paste text to extract important clauses and review potential risks.")

    input_option = st.radio("Choose input type", ["Upload file", "Enter text"], horizontal=True)
    uploaded_file = st.file_uploader("Select a contract file", type=["pdf", "docx", "txt"], key="uploaded_file")
    input_text = st.text_area("Paste contract text", key="contract_text", height=220)

    if input_option == "Upload file":
        st.info("Upload a document file to analyze its content.")
        selected_text = ""
    else:
        st.info("Paste raw contract text, then analyze it.")
        uploaded_file = None
        selected_text = input_text.strip()

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_clicked = st.button("Analyze Document")
    with col2:
        clear_clicked = st.button("Remove file / Clear text")

    if clear_clicked:
        st.session_state.uploaded_file = None
        st.session_state.contract_text = ""
        st.experimental_rerun()

    if analyze_clicked:
        if input_option == "Upload file" and uploaded_file is None:
            st.error("Please upload a contract file first.")
        elif input_option == "Enter text" and not selected_text:
            st.error("Please enter contract text to analyze.")
        else:
            with st.spinner("Uploading and analyzing document..."):
                files = {}
                data = {}
                if uploaded_file is not None:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                else:
                    data = {"text": selected_text}
                try:
                    response = requests.post(api_url, files=files, data=data, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Analysis complete.")
                        clauses = result.get("clauses") or []
                        if clauses:
                            for index, clause in enumerate(clauses, start=1):
                                st.markdown(f"### {index}. {clause.get('label', 'Unknown clause')}")
                                st.write(f"**Confidence:** {clause.get('confidence', 0.0):.2f}")
                                st.write(clause.get('text', 'No text extracted.'))
                                top_k = clause.get("top_k")
                                if top_k:
                                    with st.expander("Top predictions"):
                                        for prediction in top_k:
                                            st.write(f"- {prediction.get('label')} ({prediction.get('score', 0.0):.2f})")
                                st.write("---")
                        else:
                            st.info("No clauses were identified in the uploaded document.")
                    else:
                        st.error(f"Analysis failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the API: {e}")
