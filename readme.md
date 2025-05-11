
<div align="center">
    <h1>
    Government Scheme QnA using RAG
    </h1>
    <p>
    Personal Project • Powered by Hugging Face, LangChain & Streamlit
    </p>

<a href="https://www.python.org/downloads/release/python-3110/">
    <img src="https://img.shields.io/badge/Python-3.11+-orange" alt="Python">
</a>
<a href="https://pypi.org/project/streamlit/">
    <img src="https://img.shields.io/badge/Streamlit-1.30+-green" alt="Streamlit">
</a>
<a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">
    <img src="https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-blue" alt="Embeddings">
</a>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1">
    <img src="https://img.shields.io/badge/LLM-Mistral--7B-purple" alt="LLM">
</a>

</div>

---

## 📖 Overview

This project is a **Retrieval-Augmented Generation (RAG)** based intelligent QnA system that answers questions related to **Indian Government Schemes** by retrieving relevant policy documents and generating precise, context-rich answers using a large language model.

---

## ✨ Key Features

- 🔍 **Semantic Search** using FAISS + MiniLM embeddings  
- 🧠 **LLM-powered Answers** with `google/gemini-1.5-flash`
- 📄 Supports `.json` and `.txt` based knowledge bases
- 🌐 **Streamlit Interface** for interactive Q&A
- 🔑 Hugging Face **API Key authentication** for secure model inference

---

<table align="center">
  <tr>
    <td align="center"><b>Streamlit Interface (insert screenshot here)</b><br><img src="Streamlit Interface.png" width="80%" /></td>
  </tr>
  <tr>
    <td align="center"><b>RAG Pipeline Overview</b><br><img src="Pipeline.png" width="80%" /></td>
  </tr>
</table>

---

## ⚙️ Installation & Setup

### 📁 Clone the Repository
```bash
git clone https://github.com/theUtkarshRaj/Government-Scheme-QnA-using-RAG
cd government-scheme-rag-qa
```

### 🐍 Create and Activate Virtual Environment
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 📂 Add Knowledge Base Files
- Add `.txt` files inside the `/data` folder.
- `scheme_data.json` is already provided and used in the current setup.

---

## ▶️ Run the App

```bash
streamlit run main.py
```

> Go to [http://localhost:8501](http://localhost:8501) if the browser doesn't open automatically.

---

## 🔐 API Key Setup

After launching, enter your **Hugging Face API key** in the Streamlit sidebar.

**Steps:**

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Generate a new token (if not already created)
3. Paste it in the Streamlit input field

---

## 🧠 Models Used

| Component        | Model Name(s) & Provider                                                                 | Description                                 |
|------------------|----------------------------------------------------------------------------------------|---------------------------------------------|
| Embeddings       | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)     | Lightweight sentence transformer for retrieval |
| Language Model   | **Primary (Recommended):**<br> [`Gemini Flash`](https://aistudio.google.com/app/apikey) (Google AI Studio API, free tier available)<br><br>**Alternative (API):**<br> [`bigscience/bloomz-560m`](https://huggingface.co/bigscience/bloomz-560m) (Hugging Face Inference API, may require paid account)<br><br>**Alternative (Local):**<br> Any supported text2text-generation model via Hugging Face Transformers (e.g., Flan-T5, Llama, Gemma, etc.) | LLM for answer generation. Gemini Flash is recommended for free, reliable API access. Most Hugging Face models are not available for free API use as of 2024. |

- **Note:**
    - Gemini Flash (Google) is the recommended free API for answer generation. You need a Google AI Studio API key.
    - Most Hugging Face models (e.g., flan-t5-small, mt0-small, bloomz-560m) are no longer available for free API inference. If you use Hugging Face, check model availability and your account limits.
    - For local inference, you can use any model supported by Hugging Face Transformers, but you need sufficient hardware resources.

---

## 💬 Sample Queries

> "What benefits are provided to women in this scheme?"  
> "Tell me about the financial support available under XYZ scheme."  
> "Is there any age restriction in this policy?"

---

<!-- ## 📽️ Demo Video

[Click here to watch demo](https://drive.google.com/file/d/1R_av1NY1NSAGPCzrlX_lEw7_n7HkPl-m/view?usp=sharing)

--- -->

## 👥 Team

| Name            | Role                  |
|----------------|-----------------------|
| [Amit Anand](https://github.com/amitanand983) | Complete Project |

---

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/)
- [SentenceTransformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 📜 License

This project is licensed under the **MIT License** – use freely with attribution!