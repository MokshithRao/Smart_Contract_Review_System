import streamlit as st
import requests


CUSTOM_CSS = """
<style>
    /* Main Background */
    .stApp {
        background-color: #0a0a0f;
        background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #0a0a0f 70%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Hide default header/footer */
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* Titles */
    h1 {
        background: -webkit-linear-gradient(45deg, #a855f7, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
        margin-bottom: 1rem !important;
    }
    h2, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    /* Upload box / Text area */
    .stTextArea textarea, .stFileUploader {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(12px) !important;
        color: #f8fafc !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.3) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(168, 85, 247, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.5) !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        color: #cbd5e1 !important;
    }
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Code blocks */
    pre {
        background: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
    }
    code {
        color: #38bdf8 !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(148, 163, 184, 0.1) !important;
        margin: 2rem 0 !important;
    }
    
    /* Alerts */
    .stAlert {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        color: #e2e8f0 !important;
    }
</style>
"""

def render_streamlit_app(api_url: str = "http://127.0.0.1:8000/analyze") -> None:
    st.set_page_config(page_title="Smart Contract Review", page_icon="🧠", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Smart Contract Review System")
    st.write("Upload a contract file or paste text to extract important clauses and review potential risks.")

    input_option = st.radio("Choose input type", ["Upload file", "Enter text"], horizontal=True)

    uploaded_file = None
    selected_text = ""

    if input_option == "Upload file":
        uploaded_file = st.file_uploader("Select a contract file", type=["pdf", "docx", "txt"], key="uploaded_file")
        st.info("Upload a document file to analyze its content.")
    else:
        input_text = st.text_area("Paste contract text", key="contract_text", height=220)
        st.info("Paste raw contract text, then analyze it.")
        selected_text = input_text.strip() if input_text else ""

    analyze_clicked = st.button("Analyze Document")

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
                    response = requests.post(api_url, files=files, data=data, timeout=300)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Analysis complete.")
                        clauses = result.get("clauses") or []
                        rag_review = result.get("rag_review") or []

                        if clauses:
                            st.subheader("Important Clauses and Risk Signals")
                            for index, clause in enumerate(clauses, start=1):
                                risk_color = "#ef4444" if clause.get("risk") == "HIGH" else "#f59e0b"
                                bg_color = "rgba(239, 68, 68, 0.05)" if clause.get("risk") == "HIGH" else "rgba(245, 158, 11, 0.05)"
                                
                                card_html = f"""
                                <div style="background: {bg_color}; border: 1px solid rgba(255,255,255,0.05); border-left: 4px solid {risk_color}; padding: 20px; border-radius: 12px; margin-bottom: 16px; backdrop-filter: blur(10px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
                                    <h3 style="margin-top: 0; color: #f8fafc; font-size: 1.25rem;">{index}. {str(clause.get('label', 'Unknown clause')).title()}</h3>
                                    <div style="display: flex; gap: 10px; margin-bottom: 12px; align-items: center;">
                                        <span style="background: {risk_color}20; border: 1px solid {risk_color}40; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; font-weight: 700; color: {risk_color}; letter-spacing: 0.5px;">
                                            {str(clause.get('risk', 'UNKNOWN'))} RISK
                                        </span>
                                        <span style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.2); padding: 4px 10px; border-radius: 20px; font-size: 0.85em; color: #cbd5e1; font-weight: 600;">
                                            Confidence: {clause.get('confidence', 0.0):.2f}
                                        </span>
                                    </div>
                                    <div style="color: #cbd5e1; font-size: 0.95em; line-height: 1.5; margin-bottom: 16px;">
                                        <strong style="color: #f8fafc;">Risk Detail:</strong> <span style="opacity: 0.9;">{clause.get('reason', 'N/A')}</span>
                                    </div>
                                    <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); color: #f1f5f9; font-size: 0.95em; line-height: 1.6; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.1);">
                                        {clause.get('text', 'No text extracted.')}
                                    </div>
                                </div>
                                """
                                st.markdown(card_html, unsafe_allow_html=True)
                                top_k = clause.get("top_k")
                                if top_k:
                                    with st.expander("Top clause predictions"):
                                        for prediction in top_k:
                                            st.write(f"- {prediction.get('label')} ({prediction.get('score', 0.0):.2f})")
                                
                                if rag_review and len(rag_review) >= index:
                                    item = rag_review[index - 1]
                                    if item.get('rewrite'):
                                        with st.expander("RAG Review Suggestion - Rewrite"):
                                            st.markdown("**Suggested rewrite:**")
                                            st.code(item.get('rewrite', 'No rewrite available'))
                                            st.markdown("**Explanation:**")
                                            st.write(item.get('explanation', ''))
                                            similar_clauses = item.get('similar_clauses') or []
                                            if similar_clauses:
                                                st.markdown("**Reference clauses:**")
                                                for clause_example in similar_clauses:
                                                    st.write(f"- {clause_example.get('clause', '')} ({clause_example.get('label', '')})")

                                st.write("---")
                        else:
                            st.info("No important clauses were identified in the uploaded document.")

                        rag_status = result.get("rag_status")
                        if not rag_review:
                            if rag_status == "missing_api_token":
                                st.info("No RAG suggestions were generated because HF_API_TOKEN is not set. Please add your token to .env or environment variables.")
                            elif rag_status == "failed":
                                st.info("RAG review was attempted but could not complete. Check the backend logs for details.")
                            else:
                                st.info("No RAG suggestions were generated for this contract.")
                    else:
                        st.error(f"Analysis failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the API: {e}")


if __name__ == "__main__":
    render_streamlit_app()

