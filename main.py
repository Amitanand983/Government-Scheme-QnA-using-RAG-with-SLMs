# Remove dark mode toggle and all related code
import streamlit as st
from rag import GovernmentSchemeRAG

@st.cache_resource
def load_rag_system(json_path, hf_token, google_api_key):
    return GovernmentSchemeRAG(json_path, hf_token, google_api_key)

def main():
    st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
    st.title("🗂️ Government Scheme QnA")
    st.markdown("Ask a question about Indian government schemes. Example: _What schemes are available for women entrepreneurs?_\n")

    # Sidebar for upload and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key Input (Takes precedence, must be entered first)
        st.subheader("🔑 API Key")
        hf_token = st.text_input("Hugging Face Token", type="password", placeholder="Enter Hugging Face API Token")
        google_api_key = st.text_input("Google Gemini API Key", type="password", placeholder="Enter Google Gemini API Key")
        if not hf_token and not google_api_key:
            st.warning("Please enter your Hugging Face API Token or Google Gemini API Key to proceed.")
            st.stop()  # Stop execution until at least one API key is provided

        # File uploader
        uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
        if uploaded_file:
            st.session_state.json_path = uploaded_file
        else:
            st.session_state.json_path = "scheme_data.json"

        # History section at the end of the sidebar
        st.markdown("---")
        st.subheader("📜 History")
        if "history" not in st.session_state:
            st.session_state.history = []
        for entry in reversed(st.session_state.history):
            with st.expander(f"❓ {entry['question'][:40]}..."):
                st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

    # Load RAG system only after API key is provided
    rag_system = load_rag_system(st.session_state.json_path, hf_token, google_api_key)
    st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

    # Example input section
    st.subheader("💡 Ask Your Question")
    example_queries = [
        "What schemes are available for women entrepreneurs?",
        "Schemes related to education for girls?",
        "Financial assistance for farmers?",
        "Startup schemes in India?"
    ]

    selected_example = st.selectbox("📌 Popular Queries:", [""] + example_queries)
    user_query = st.text_input("🔍 Type your question here:", value=selected_example)

    # Filter by ministry
    all_ministries = sorted(set([meta.get("ministry", "Unknown") for meta in rag_system.metadata]))
    selected_ministry = st.selectbox("🏛️ Filter by Ministry", ["All"] + all_ministries)

    # Process query
    if user_query.strip():
        with st.spinner("🤔 Thinking..."):
            results = rag_system.query(user_query, top_k=3)
            if selected_ministry != "All":
                results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
            context = "\n\n".join([r["chunk"] for r in results])
            generated_answer = rag_system.generate_answer(user_query, context)

        # Save to history
        st.session_state.history.append({
            "question": user_query,
            "answer": generated_answer,
            "sources": results
        })

    # Show latest answer
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.subheader("🧠 Answer")
        st.write(latest["answer"])

        # Feedback
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", key="like"):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 Not helpful", key="dislike"):
                st.warning("We'll use this to improve.")

        # Sources
        st.subheader("📄 Sources")
        for idx, result in enumerate(latest["sources"], 1):
            meta = result["metadata"]
            title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
            with st.expander(f"Source {idx}: {title}"):
                st.markdown(result["chunk"])

    # Show previous history
    if len(st.session_state.history) > 1:
        with st.expander("📜 Previous Questions"):
            for entry in reversed(st.session_state.history[:-1]):
                st.markdown(f"**❓ Q:** {entry['question']}")
                st.markdown(f"**💡 A:** {entry['answer']}")
                st.markdown("---")

if __name__ == "__main__":
    main()