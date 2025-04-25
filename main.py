import streamlit as st
from rag import GovernmentSchemeRAG

@st.cache_resource
def load_rag_system(json_path, hf_token):
    return GovernmentSchemeRAG(json_path, hf_token)

def main():
    # Set page configuration
    st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
    st.title("🗂️ Government Scheme QnA")
    st.markdown("Ask a question about Indian government schemes. Example: _What schemes are available for women entrepreneurs?_")

    # Sidebar for upload and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key Input (Takes precedence, must be entered first)
        st.subheader("🔑 API Key")
        hf_token = st.text_input("Hugging Face Token", type="password", placeholder="Enter Hugging Face API Token")
        if not hf_token:
            st.warning("Please enter your Hugging Face API Token to proceed.")
            st.stop()  # Stop execution until API key is provided

        # File uploader
        uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
        if uploaded_file:
            st.session_state.json_path = uploaded_file
        else:
            st.session_state.json_path = "scheme_data.json"

        # Theme toggle
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False  # Default to light mode

        if st.button("🌙 Toggle Dark Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            js_code = """
                const root = document.querySelector('html');
                if (root.classList.contains('dark')) {
                    root.classList.remove('dark');
                } else {
                    root.classList.add('dark');
                }
            """
            st.components.v1.html(f"<script>{js_code}</script>", height=0)

        # History view
        st.markdown("---")
        st.subheader("📜 History")
        if "history" not in st.session_state:
            st.session_state.history = []
        for entry in reversed(st.session_state.history):
            with st.expander(f"❓ {entry['question'][:40]}..."):
                st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

    # Load RAG system only after API key is provided
    rag_system = load_rag_system(st.session_state.json_path, hf_token)
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
                st.warning("We’ll use this to improve.")

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


# ### Version -4

# import streamlit as st
# from rag import GovernmentSchemeRAG

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")
#     st.markdown("Ask a question about Indian government schemes. Example: _What schemes are available for women entrepreneurs?_")

#     # Sidebar for upload and configuration
#     with st.sidebar:
#         st.header("⚙️ Configuration")

#         # File uploader
#         uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#         else:
#             st.session_state.json_path = "scheme_data.json"

#         # Theme toggle
#         if "dark_mode" not in st.session_state:
#             st.session_state.dark_mode = False  # Default to light mode

#         if st.button("🌙 Toggle Dark Mode"):
#             st.session_state.dark_mode = not st.session_state.dark_mode
#             js_code = """
#                 const root = document.querySelector('html');
#                 if (root.classList.contains('dark')) {
#                     root.classList.remove('dark');
#                 } else {
#                     root.classList.add('dark');
#                 }
#             """
#             st.components.v1.html(f"<script>{js_code}</script>", height=0)

#         # History view
#         st.markdown("---")
#         st.subheader("📜 History")
#         if "history" not in st.session_state:
#             st.session_state.history = []
#         for entry in reversed(st.session_state.history):
#             with st.expander(f"❓ {entry['question'][:40]}..."):
#                 st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

#     # Load RAG system
#     rag_system = load_rag_system(st.session_state.json_path)
#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     # Example input section
#     st.subheader("💡 Ask Your Question")
#     example_queries = [
#         "What schemes are available for women entrepreneurs?",
#         "Schemes related to education for girls?",
#         "Financial assistance for farmers?",
#         "Startup schemes in India?"
#     ]

#     selected_example = st.selectbox("📌 Popular Queries:", [""] + example_queries)
#     user_query = st.text_input("🔍 Type your question here:", value=selected_example)

#     # Filter by ministry
#     all_ministries = sorted(set([meta.get("ministry", "Unknown") for meta in rag_system.metadata]))
#     selected_ministry = st.selectbox("🏛️ Filter by Ministry", ["All"] + all_ministries)

#     # Process query
#     if user_query.strip():
#         with st.spinner("🤔 Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             if selected_ministry != "All":
#                 results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         # Save to history
#         st.session_state.history.append({
#             "question": user_query,
#             "answer": generated_answer,
#             "sources": results
#         })

#     # Show latest answer
#     if st.session_state.history:
#         latest = st.session_state.history[-1]
#         st.subheader("🧠 Answer")
#         st.write(latest["answer"])

#         # Feedback
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("👍 Helpful", key="like"):
#                 st.success("Thanks for your feedback!")
#         with col2:
#             if st.button("👎 Not helpful", key="dislike"):
#                 st.warning("We’ll use this to improve.")

#         # Sources
#         st.subheader("📄 Sources")
#         for idx, result in enumerate(latest["sources"], 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])

#     # Show previous history
#     if len(st.session_state.history) > 1:
#         with st.expander("📜 Previous Questions"):
#             for entry in reversed(st.session_state.history[:-1]):
#                 st.markdown(f"**❓ Q:** {entry['question']}")
#                 st.markdown(f"**💡 A:** {entry['answer']}")
#                 st.markdown("---")

# if __name__ == "__main__":
#     main()


# # Version -4 compatible with ver-3 of rag.py

# import streamlit as st
# from rag import GovernmentSchemeRAG
# from streamlit_webrtc import webrtc_streamer, WebRtcMode
# import speech_recognition as sr
# import av
# from io import BytesIO

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def transcribe_audio(audio):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio) as source:
#         audio_data = recognizer.record(source)
#         return recognizer.recognize_google(audio_data)

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")

#     # Sidebar config
#     with st.sidebar:
#         st.header("⚙️ Configuration")

#         # Theme toggle button
#         if "dark_mode" not in st.session_state:
#             st.session_state.dark_mode = False  # Default to light mode

#         if st.button("🌙 Toggle Dark Mode"):
#             st.session_state.dark_mode = not st.session_state.dark_mode

#         # Apply theme based on session state
#         if st.session_state.dark_mode:
#             st.markdown("""
#                 <style>
#                     body { background-color: #0e1117; color: #fafafa; }
#                 </style>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#                 <style>
#                     body { background-color: #ffffff; color: #000000; }
#                 </style>
#             """, unsafe_allow_html=True)

#         # File upload
#         uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#         else:
#             st.session_state.json_path = "scheme_data.json"

#         # History
#         st.markdown("---")
#         st.subheader("📜 History")
#         if "history" not in st.session_state:
#             st.session_state.history = []
#         for entry in reversed(st.session_state.history):
#             with st.expander(f"❓ {entry['question'][:40]}..."):
#                 st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

#     # Load RAG system
#     rag_system = load_rag_system(st.session_state.json_path)
#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     # Unified Input Section
#     st.subheader("💡 Ask Your Question")
#     example_queries = [
#         "What schemes are available for women entrepreneurs?",
#         "Schemes related to education for girls?",
#         "Financial assistance for farmers?",
#         "Startup schemes in India?"
#     ]

#     # Dropdown for suggestions
#     selected_example = st.selectbox("📌 Popular Queries:", [""] + example_queries)

#     # Text input field for typing or displaying selected suggestion/audio transcription
#     user_query = st.text_input("🔍 Type your question here:", value=selected_example)

#     # Voice input integration
#     st.markdown("🎙️ Use voice input (optional):")
#     voice_text = None
#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=256,
#         media_stream_constraints={"audio": True, "video": False},
#         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     )

#     if webrtc_ctx.audio_receiver:
#         try:
#             audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             wav_buffer = BytesIO()
#             for frame in audio_frames:
#                 sound = av.AudioFrame.from_ndarray(frame.to_ndarray(), layout="mono")
#                 sound.sample_rate = frame.sample_rate
#                 sound.to_audio().export(wav_buffer, format="wav")
#             wav_buffer.seek(0)
#             voice_text = transcribe_audio(wav_buffer)
#             st.success(f"🗣️ Transcribed: {voice_text}")
#             user_query = voice_text  # Replace the text input with transcribed text
#         except Exception as e:
#             st.error(f"Error during transcription: {e}")

#     # Scheme filter
#     all_ministries = sorted(set([meta.get("ministry", "Unknown") for meta in rag_system.metadata]))
#     selected_ministry = st.selectbox("🏛️ Filter by Ministry", ["All"] + all_ministries)

#     # Process the query
#     if user_query.strip():  # Ensure the query is not empty
#         with st.spinner("🤔 Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             if selected_ministry != "All":
#                 results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         # Save to history
#         st.session_state.history.append({
#             "question": user_query,
#             "answer": generated_answer,
#             "sources": results
#         })

#     # Show latest answer
#     if st.session_state.history:
#         latest = st.session_state.history[-1]
#         st.subheader("🧠 Answer")
#         st.write(latest["answer"])

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("👍 Helpful", key="like"):
#                 st.success("Thanks for your feedback!")
#         with col2:
#             if st.button("👎 Not helpful", key="dislike"):
#                 st.warning("We’ll use this to improve.")

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(latest["sources"], 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])

# if __name__ == "__main__":
#     main()

# # # ### Improved version -3 

# import streamlit as st
# from rag import GovernmentSchemeRAG
# from streamlit_webrtc import webrtc_streamer, WebRtcMode
# import speech_recognition as sr
# import av
# from io import BytesIO

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def transcribe_audio(audio):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio) as source:
#         audio_data = recognizer.record(source)
#         return recognizer.recognize_google(audio_data)

# def main():
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")

#     # Sidebar config
#     with st.sidebar:
#         st.header("⚙️ Configuration")

#         uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#         else:
#             st.session_state.json_path = "scheme_data.json"

#         dark_mode = st.checkbox("🌙 Dark Mode (manual)", value=False)
#         if dark_mode:
#             st.markdown("<style>body { background-color: #0e1117; color: #fafafa; }</style>", unsafe_allow_html=True)

#         # History
#         st.markdown("---")
#         st.subheader("📜 History")
#         if "history" not in st.session_state:
#             st.session_state.history = []
#         for entry in reversed(st.session_state.history):
#             with st.expander(f"❓ {entry['question'][:40]}..."):
#                 st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

#     # Load RAG system
#     rag_system = load_rag_system(st.session_state.json_path)
#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     # Popular queries
#     st.subheader("💡 Suggested Questions")
#     example_queries = [
#         "What schemes are available for women entrepreneurs?",
#         "Schemes related to education for girls?",
#         "Financial assistance for farmers?",
#         "Startup schemes in India?"
#     ]
#     selected_example = st.selectbox("📌 Pick a popular query or type your own:", [""] + example_queries)

#     # Voice input
#     st.markdown("🎙️ Use voice input (optional):")
#     voice_text = None
#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=256,
#         media_stream_constraints={"audio": True, "video": False},
#         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     )

#     if webrtc_ctx.audio_receiver:
#         try:
#             audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             wav_buffer = BytesIO()
#             for frame in audio_frames:
#                 sound = av.AudioFrame.from_ndarray(frame.to_ndarray(), layout="mono")
#                 sound.sample_rate = frame.sample_rate
#                 sound.to_audio().export(wav_buffer, format="wav")
#             wav_buffer.seek(0)
#             voice_text = transcribe_audio(wav_buffer)
#             st.success(f"🗣️ Transcribed: {voice_text}")
#         except:
#             pass

#     # Scheme filter
#     all_ministries = sorted(set([meta.get("ministry", "Unknown") for meta in rag_system.metadata]))
#     selected_ministry = st.selectbox("🏛️ Filter by Ministry", ["All"] + all_ministries)

#     # Determine query
#     user_query = voice_text or selected_example or st.text_input("🔍 Or type your own question:")

#     if user_query:
#         with st.spinner("🤔 Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             if selected_ministry != "All":
#                 results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         # Save to history
#         st.session_state.history.append({
#             "question": user_query,
#             "answer": generated_answer,
#             "sources": results
#         })

#     # Show latest answer
#     if st.session_state.history:
#         latest = st.session_state.history[-1]
#         st.subheader("🧠 Answer")
#         st.write(latest["answer"])

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("👍 Helpful", key="like"):
#                 st.success("Thanks for your feedback!")
#         with col2:
#             if st.button("👎 Not helpful", key="dislike"):
#                 st.warning("We’ll use this to improve.")

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(latest["sources"], 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])

# if __name__ == "__main__":
#     main()


# ### Imporved UI and More features
# import streamlit as st
# from rag import GovernmentSchemeRAG

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def main():
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")
#     st.markdown("Ask a question about Indian government schemes. Example: _What schemes are available for women entrepreneurs?_")

#     # Sidebar for upload
#     with st.sidebar:
#         st.header("⚙️ Configuration")
#         uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#         else:
#             st.session_state.json_path = "scheme_data.json"

#     # Load RAG system
#     rag_system = load_rag_system(st.session_state.json_path)
#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     # Initialize history
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     # User input
#     user_query = st.text_input("🔍 Enter your question:")
#     if user_query:
#         with st.spinner("🤔 Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         # Save to history
#         st.session_state.history.append({
#             "question": user_query,
#             "answer": generated_answer,
#             "sources": results
#         })

#     # Show latest answer
#     if st.session_state.history:
#         latest = st.session_state.history[-1]
#         st.subheader("🧠 Answer")
#         st.write(latest["answer"])

#         # Feedback
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("👍 Helpful", key="like"):
#                 st.success("Thanks for your feedback!")
#         with col2:
#             if st.button("👎 Not helpful", key="dislike"):
#                 st.warning("We’ll use this to improve.")

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(latest["sources"], 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])

#     # Show past history
#     if len(st.session_state.history) > 1:
#         with st.expander("📜 Previous Questions"):
#             for entry in reversed(st.session_state.history[:-1]):
#                 st.markdown(f"**❓ Q:** {entry['question']}")
#                 st.markdown(f"**💡 A:** {entry['answer']}")
#                 st.markdown("---")

# if __name__ == "__main__":
#     main()

### Working Code -2(Giving Ans with sources -1)

# import streamlit as st
# from rag import GovernmentSchemeRAG


# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)


# def main():
#     st.set_page_config(page_title="Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")
#     st.write("Ask a question about Indian government schemes. Example: _What schemes are available for women entrepreneurs?_")

#     # Set default path
#     if "json_path" not in st.session_state:
#         st.session_state.json_path = "scheme_data.json"

#     rag_system = load_rag_system(st.session_state.json_path)

#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     user_query = st.text_input("🔍 Enter your question:")
#     if user_query:
#         with st.spinner("Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         st.subheader("🧠 Answer")
#         st.write(generated_answer)

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(results, 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])


# if __name__ == "__main__":
#     main()



## Working Code -1

# # main.py - Streamlit interface for the RAG application
# import streamlit as st
# import os
# import json
# import tempfile
# from rag import GovernmentSchemeRAG

# st.set_page_config(
#     page_title="Government Scheme Q&A",
#     page_icon="🏛️",
#     layout="wide"
# )

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def main():
#     st.title("🏛️ Government Scheme Question & Answer System")
#     st.write("Ask questions about government schemes from MyScheme portal")
    
#     # File upload section (first run only)
#     if 'data_loaded' not in st.session_state:
#         st.session_state.data_loaded = False
    
#     if not st.session_state.data_loaded:
#         # Option to use local file path
#         use_local_file = st.checkbox("Use local JSON file")
        
#         if use_local_file:
#             file_path = "/Users/amitanand/Desktop/Utkarsh/scheme_data.json"
#             st.text(f"Using local file: {file_path}")
            
#             if st.button("Load Data"):
#                 st.session_state.json_path = file_path
#                 st.session_state.data_loaded = True
#                 st.experimental_rerun()
#         else:
#             uploaded_file = st.file_uploader("Upload your schemes JSON file", type=["json"])
            
#             if uploaded_file:
#                 # Save the uploaded file to a temporary file
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
#                     tmp_file.write(uploaded_file.getvalue())
#                     temp_json_path = tmp_file.name
                
#                 st.session_state.json_path = temp_json_path
#                 st.session_state.data_loaded = True
#                 st.experimental_rerun()
    
#     # Once data is loaded, show the QA interface
#     if st.session_state.data_loaded:
#         with st.spinner("Loading RAG system... This may take a while for large JSON files."):
#             # Initialize RAG system
#             rag_system = load_rag_system(st.session_state.json_path)
        
#         # Check if RAG system was initialized successfully
#         if rag_system and hasattr(rag_system, 'chunks') and len(rag_system.chunks) > 0:
#             st.success(f"Successfully loaded {len(rag_system.chunks)} chunks from {len(rag_system.schemes_data)} schemes!")
            
#             # Input for user question
#             question = st.text_input("Ask a question about government schemes:", 
#                                     "What schemes are available for farmers in Maharashtra?")
            
#             # Number of sources to retrieve
#             top_k = st.slider("Number of sources to retrieve:", min_value=1, max_value=5, value=3)
            
#             # Process question on button click
#             if st.button("Get Answer"):
#                 with st.spinner("Searching and generating answer..."):
#                     result = rag_system.query(question, top_k=top_k)
                
#                 # Display answer
#                 st.subheader("Answer")
#                 st.write(result)
                
#                 # Display sources
#                 st.subheader("Sources")
#                 for idx, source in enumerate(result, 1):
#                     with st.expander(f"Source {idx}: {source.get('scheme_name', 'Unknown Scheme')}"):
#                         st.write(f"**Ministry:** {source.get('ministry', 'N/A')}")
#                         st.write(f"**Department:** {source.get('department', 'N/A')}")
#                         st.write(f"**Page:** {source.get('page', 'N/A')}")
#                         if 'type' in source:
#                             st.write(f"**Content Type:** {source['type']}")
#         else:
#             st.error("Failed to initialize RAG system. Please check the JSON file and try again.")

# if __name__ == "__main__":
#     main()