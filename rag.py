### APi key input 
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import re
import google.generativeai as genai

class GovernmentSchemeRAG:
    def __init__(self, json_path, hf_token="", google_api_key=""):
        self.json_path = json_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.dimension = None
        self.hf_token = hf_token
        self.google_api_key = google_api_key
        self.chunks, self.metadata = self.chunk_documents()
        if not self.chunks:
            raise ValueError("No chunks available to create embeddings.")
        self.create_index()

    def chunk_documents(self):
        chunks = []
        metadata = []
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:  # Specify encoding
                self.schemes_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.json_path}")
            return [], []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.json_path}. Check file format.")
            return [], []
        except Exception as e:
            print(f"An unexpected error occurred loading the JSON: {e}")
            return [], []

        for scheme in self.schemes_data:
            data = scheme.get("data", {})

            text_parts = []
            scheme_name = data.get("scheme_name", "Unknown Scheme")
            ministry = data.get("ministry", "Unknown Ministry")
            department = data.get("department", "Unknown Department")

            text_parts.append(f"Scheme: {scheme_name}")
            text_parts.append(f"Ministry: {ministry}")
            text_parts.append(f"Department: {department}")

            for key in ["details_content", "eligibility_content", "application_process"]:
                content = data.get(key, [])
                if isinstance(content, list):
                    # Clean up potential None values or non-string items if necessary
                    cleaned_content = [str(item) for item in content if item is not None]
                    text_parts.extend(cleaned_content)
                elif content is not None:  # Handle cases where it might be a single string
                    text_parts.append(str(content))

            chunk = "\n".join(text_parts).strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({
                    "scheme_name": scheme_name,
                    "ministry": ministry,
                    "department": department
                })

        return chunks, metadata

    def create_index(self):
        if not self.chunks:
            print("Skipping index creation as no chunks were loaded.")
            return
        embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks]).astype('float32')  # Ensure float32

        if embeddings.ndim == 1:
            if embeddings.shape[0] > 0:  # Check if the single dimension is not empty
                self.dimension = embeddings.shape[0]
                embeddings = embeddings.reshape(1, -1)
            else:
                print("Warning: Embeddings array is empty or invalid.")
                return  # Cannot create index with empty embeddings
        elif embeddings.shape[0] == 0:  # Check if the 2D array has no rows
            print("Warning: Embeddings array is empty.")
            return  # Cannot create index with empty embeddings
        else:
            self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        print(f"FAISS index created successfully with {self.index.ntotal} vectors.")

    def query(self, question, top_k=3):
        if not self.index or self.index.ntotal == 0:
            return []  # Return empty if index doesn't exist or is empty
        question_embedding = self.embedding_model.encode(question).reshape(1, -1).astype('float32')  # Ensure float32
        distances, indices = self.index.search(question_embedding, top_k)

        results = []
        for i in indices[0]:
            # Check index bounds robustly
            if 0 <= i < len(self.chunks):
                results.append({
                    "chunk": self.chunks[i],
                    "metadata": self.metadata[i]
                })
            else:
                print(f"Warning: Index {i} out of bounds for chunks list (length {len(self.chunks)}).")
        return results

    def generate_answer(self, question, context):
        prompt = f"""
You are an expert assistant for Indian government schemes.

Context:
{context}

User Question:
{question}

Instructions:
1. Search the context for a scheme whose name or description exactly matches what the user asked for.
2. If you find an exact match, provide a detailed answer ONLY about that scheme, including:
   - **Scheme Name**
   - **Ministry/Department**
   - **Purpose**
   - **Eligibility**
   - **Benefits/Assistance**
   - **Application Process** (with steps if available)
   - **Official Website Link** (if found)
   - Use clear sections with bold headers (Markdown: **Header:**).
3. If you do NOT find an exact match:
   - Clearly state: "No exact match found for your query."
   - List the names of the most relevant or related schemes (if any), but DO NOT provide their details.
   - Example: "Related schemes: Scheme A, Scheme B, Scheme C."
4. Never provide details for unrelated or only partially matching schemes.
5. Be concise, clear, and use Markdown formatting for readability.

Remember: Only answer about the exact scheme if found. If not, just list related scheme names, no details.
"""
        answer = "Could not generate answer using any model."  # Default error message

        # Try Gemini Flash if Google API key is provided
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                response = model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                answer = f"Error: Could not connect to Gemini API - {e}"
        # Otherwise, try Hugging Face API
        elif self.hf_token:
            api_url = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"
            headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
            payload = {
                "inputs": prompt,
                "options": {
                    "wait_for_model": True,
                    "max_length": 1000,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                output = response.json()
                if output and isinstance(output, list) and 'generated_text' in output[0]:
                    answer = output[0].get("generated_text", "No answer returned by model.")
                else:
                    answer = f"Unexpected response format from API: {output}"
            except requests.exceptions.RequestException as e:
                print(f"Error calling Hugging Face API: {e}")
                answer = f"Error: Could not connect to Hugging Face API - {e}"
            except Exception as e:
                print(f"Error processing Hugging Face response: {e}")
                answer = f"Error processing Hugging Face response: {e}"
        else:
            answer = "No API key provided for Gemini or Hugging Face."

        # Post-processing for better formatting
        headers = [
            "Scheme Name:", "Ministry/Department:", "Purpose:", "Eligibility:",
            "Benefits:", "Key Benefits:", "Application Process:", "Application Steps:",
            "Required Documents:", "Website Link:", "Official Link:", "Source:",
            "Overview:", "Relevant Schemes:", "Scheme Details:"
        ]
        for header in headers:
            answer = answer.replace(header, f"**{header}**")

        # Format application steps
        if "**Application Process:**" in answer or "**Application Steps:**" in answer:
            lines = answer.split('\n')
            formatted_lines = []
            in_steps = False
            step_count = 0
            for line in lines:
                if line.strip().startswith("**Application Process:**") or line.strip().startswith("**Application Steps:**"):
                    in_steps = True
                    formatted_lines.append(line)
                elif in_steps:
                    if re.match(r'^\s*\d+[\.\)]\s+', line.strip()):
                        step_count += 1
                        formatted_lines.append(f"\n{step_count}. {line.strip().split('.', 1)[1].strip()}")
                    elif line.strip().startswith('- '):
                        formatted_lines.append(f"\n• {line.strip()[2:]}")
                    elif line.strip().startswith('**'):
                        in_steps = False
                        formatted_lines.append(line)
                    elif line.strip() and not line.strip().startswith('**'):
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            answer = '\n'.join(formatted_lines)

        # Make URLs clickable
        urls = re.findall(r'(https?://[^\s]+)', answer)
        for url in urls:
            if f"[{url}]({url})" not in answer:
                if f"**Website Link:** {url}" in answer:
                    answer = answer.replace(f"**Website Link:** {url}", f"**Website Link:** [{url}]({url})")
                elif f"**Official Link:** {url}" in answer:
                    answer = answer.replace(f"**Official Link:** {url}", f"**Official Link:** [{url}]({url})")
                else:
                    answer = answer.replace(url, f"[{url}]({url})")
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        return answer.strip()


# # # rag.py

# # import json
# # import numpy as np
# # import faiss
# # import os
# # import requests
# # from sentence_transformers import SentenceTransformer
# # from dotenv import load_dotenv
# # import google.generativeai as genai # Added for Gemini

# # load_dotenv()

# # class GovernmentSchemeRAG:
# #     def __init__(self, json_path):
# #         self.json_path = json_path
# #         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# #         self.index = None
# #         self.dimension = None

# #         # Load API Keys
# #         self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
# #         self.google_api_key = os.getenv("GOOGLE_API_KEY")

# #         # Configure Gemini
# #         if self.google_api_key:
# #             try:
# #                 genai.configure(api_key=self.google_api_key)
# #             except Exception as e:
# #                 print(f"Error configuring Gemini API: {e}") # Non-blocking error
# #                 self.google_api_key = None # Disable Gemini if config fails
# #         else:
# #              print("Warning: GOOGLE_API_KEY not found in .env file. Gemini model will be unavailable.")


# #         self.chunks, self.metadata = self.chunk_documents()
# #         if not self.chunks:
# #             raise ValueError("No chunks available to create embeddings.")

# #         self.create_index()

# #     def chunk_documents(self):
# #         chunks = []
# #         metadata = []
# #         try:
# #             with open(self.json_path, 'r', encoding='utf-8') as f: # Specify encoding
# #                 self.schemes_data = json.load(f)
# #         except FileNotFoundError:
# #             st.error(f"Error: JSON file not found at {self.json_path}")
# #             return [], []
# #         except json.JSONDecodeError:
# #             st.error(f"Error: Could not decode JSON from {self.json_path}. Check file format.")
# #             return [], []
# #         except Exception as e:
# #             st.error(f"An unexpected error occurred loading the JSON: {e}")
# #             return [], []


# #         for scheme in self.schemes_data:
# #             data = scheme.get("data", {})

# #             text_parts = []
# #             scheme_name = data.get("scheme_name", "Unknown Scheme")
# #             ministry = data.get("ministry", "Unknown Ministry")
# #             department = data.get("department", "Unknown Department")

# #             text_parts.append(f"Scheme: {scheme_name}")
# #             text_parts.append(f"Ministry: {ministry}")
# #             text_parts.append(f"Department: {department}")

# #             for key in ["details_content", "eligibility_content", "application_process"]:
# #                 content = data.get(key, [])
# #                 if isinstance(content, list):
# #                      # Clean up potential None values or non-string items if necessary
# #                     cleaned_content = [str(item) for item in content if item is not None]
# #                     text_parts.extend(cleaned_content)
# #                 elif content is not None: # Handle cases where it might be a single string
# #                     text_parts.append(str(content))


# #             chunk = "\n".join(text_parts).strip()
# #             if chunk:
# #                 chunks.append(chunk)
# #                 metadata.append({
# #                     "scheme_name": scheme_name,
# #                     "ministry": ministry,
# #                     "department": department
# #                 })

# #         return chunks, metadata

# #     def create_index(self):
# #         if not self.chunks:
# #             print("Skipping index creation as no chunks were loaded.")
# #             return
# #         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks]).astype('float32') # Ensure float32

# #         if embeddings.ndim == 1:
# #              if embeddings.shape[0] > 0: # Check if the single dimension is not empty
# #                 self.dimension = embeddings.shape[0]
# #                 embeddings = embeddings.reshape(1, -1)
# #              else:
# #                  print("Warning: Embeddings array is empty or invalid.")
# #                  return # Cannot create index with empty embeddings
# #         elif embeddings.shape[0] == 0: # Check if the 2D array has no rows
# #             print("Warning: Embeddings array is empty.")
# #             return # Cannot create index with empty embeddings
# #         else:
# #             self.dimension = embeddings.shape[1]


# #         self.index = faiss.IndexFlatL2(self.dimension)
# #         self.index.add(embeddings)
# #         print(f"FAISS index created successfully with {self.index.ntotal} vectors.")


# #     def query(self, question, top_k=3):
# #         if not self.index or self.index.ntotal == 0:
# #              return [] # Return empty if index doesn't exist or is empty
# #         question_embedding = self.embedding_model.encode(question).reshape(1, -1).astype('float32') # Ensure float32
# #         distances, indices = self.index.search(question_embedding, top_k)

# #         results = []
# #         for i in indices[0]:
# #             # Check index bounds robustly
# #             if 0 <= i < len(self.chunks):
# #                 results.append({
# #                     "chunk": self.chunks[i],
# #                     "metadata": self.metadata[i]
# #                 })
# #             else:
# #                  print(f"Warning: Index {i} out of bounds for chunks list (length {len(self.chunks)}).")
# #         return results

# #     # --- Updated generate_answer ---
# #     def generate_answer(self, question, context, model_choice='HuggingFace'):
# #         prompt = f"""
# # Context about government schemes:
# # {context}

# # Question: {question}

# # Given the context below about a government scheme, answer the user's question concisely, focusing on the key details requested.

# # If available, mention:
# # - Scheme Name
# # - Purpose
# # - Eligibility
# # - Key Benefits
# # - Application Process Overview (briefly)
# # - Website Link (if explicitly found in context)

# # Highlight important section titles in **bold**.
# # If information is missing for a section, simply omit that section. Be clear and direct.
# # """
# #         answer = f"Could not generate answer using {model_choice}." # Default error message

# #         if model_choice == 'Gemini' and self.google_api_key:
# #             try:
# #                 generation_config = {"temperature": 0.1, "max_output_tokens": 500} # Simplified config
# #                 safety_settings=[ # Basic safety settings
# #                     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
# #                     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
# #                     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
# #                     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
# #                 ]
# #                 model = genai.GenerativeModel(model_name="gemini-1.0-pro",
# #                                               generation_config=generation_config,
# #                                               safety_settings=safety_settings)
# #                 response = model.generate_content(prompt)
# #                 answer = response.text
# #             except Exception as e:
# #                 print(f"Error generating answer with Gemini: {e}")
# #                 answer = f"Error generating answer with Gemini: {e}"

# #         elif model_choice == 'HuggingFace' and self.hf_token:
# #             api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
# #             headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
# #             payload = {"inputs": prompt, "options": {"wait_for_model": True, "max_length": 450, "temperature": 0.1}}
# #             try:
# #                 response = requests.post(api_url, headers=headers, json=payload)
# #                 response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
# #                 output = response.json()
# #                 if output and isinstance(output, list) and 'generated_text' in output[0]:
# #                      answer = output[0].get("generated_text", "No answer returned by Flan-T5.")
# #                 else:
# #                      answer = f"Unexpected response format from Flan-T5 API: {output}"

# #             except requests.exceptions.RequestException as e:
# #                 print(f"Error calling Hugging Face API: {e}")
# #                 answer = f"Error: Could not connect to Hugging Face API - {e}"
# #             except Exception as e:
# #                 print(f"Error processing Hugging Face response: {e}")
# #                 answer = f"Error processing Hugging Face response: {e}"

# #         else:
# #              if model_choice == 'Gemini':
# #                  answer = "Gemini model unavailable (check GOOGLE_API_KEY in .env)."
# #              elif model_choice == 'HuggingFace':
# #                  answer = "Hugging Face model unavailable (check HUGGINGFACE_TOKEN in .env)."


# #         # Apply post-processing (Common for both models)
# #         answer = answer.replace("Scheme Name:", "**Scheme Name:**")
# #         answer = answer.replace("Ministry/Department:", "**Ministry/Department:**")
# #         answer = answer.replace("Purpose:", "**Purpose:**")
# #         answer = answer.replace("Benefits:", "**Benefits:**")
# #         answer = answer.replace("Key Benefits:", "**Key Benefits:**")
# #         answer = answer.replace("Eligibility:", "**Eligibility:**")
# #         answer = answer.replace("Application Process:", "**Application Process:**")
# #         answer = answer.replace("Application Process Overview:", "**Application Process Overview:**")
# #         answer = answer.replace("Required Documents:", "**Required Documents:**")
# #         answer = answer.replace("Website Link:", "**Website Link:**")
# #         answer = answer.replace("Source:", "**Source:**") # Keep this if your prompt might generate it

# #         # Make URLs clickable
# #         urls = re.findall(r'(https?://[^\s]+)', answer)
# #         for url in urls:
# #              # Basic check to avoid mangling markdown links if already formatted
# #              if f"[{url}]({url})" not in answer and f"**Website Link:** {url}" in answer :
# #                  answer = answer.replace(url, f"[{url}]({url})")

# #         # Format application steps (basic newline formatting)
# #         if "**Application Process:**" in answer or "**Application Process Overview:**" in answer:
# #             lines = answer.split('\n')
# #             formatted_lines = []
# #             in_app_process = False
# #             for line in lines:
# #                 if line.strip().startswith("**Application Process"):
# #                      in_app_process = True
# #                      formatted_lines.append(line)
# #                 elif in_app_process and re.match(r'^\s*\d+\.\s+', line.strip()):
# #                      formatted_lines.append(line.strip()) # Keep numbered steps
# #                 elif in_app_process and line.strip().startswith('- '):
# #                      formatted_lines.append(line.strip()) # Keep bullet points
# #                 elif in_app_process and line.strip() == "":
# #                     # Stop adding newlines if the section seems to end
# #                     if len(formatted_lines) > 0 and formatted_lines[-1].strip() != "":
# #                         in_app_process = False # Assume end of section on blank line
# #                     formatted_lines.append(line) # Keep blank lines within reason
# #                 elif in_app_process:
# #                      formatted_lines.append(line) # Keep other lines in the section
# #                 else:
# #                      formatted_lines.append(line) # Add lines outside the section
# #             answer = "\n".join(formatted_lines)


# #         return answer

# # ## Version -3 compatible with version -3 of main.py

# # import json
# # import numpy as np
# # import faiss
# # import os
# # import requests
# # from sentence_transformers import SentenceTransformer
# # from dotenv import load_dotenv
# # import re

# # load_dotenv()

# # class GovernmentSchemeRAG:
# #     def __init__(self, json_path):
# #         self.json_path = json_path
# #         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# #         self.index = None
# #         self.dimension = None

# #         self.chunks, self.metadata = self.chunk_documents()
# #         if not self.chunks:
# #             raise ValueError("No chunks available to create embeddings.")

# #         self.create_index()

# #     def chunk_documents(self):
# #         chunks = []
# #         metadata = []

# #         with open(self.json_path, 'r') as f:
# #             self.schemes_data = json.load(f)

# #         for scheme in self.schemes_data:
# #             data = scheme.get("data", {})

# #             text_parts = []
# #             scheme_name = data.get("scheme_name", "Unknown Scheme")
# #             ministry = data.get("ministry", "Unknown Ministry")
# #             department = data.get("department", "Unknown Department")

# #             text_parts.append(f"Scheme: {scheme_name}")
# #             text_parts.append(f"Ministry: {ministry}")
# #             text_parts.append(f"Department: {department}")

# #             for key in ["details_content", "eligibility_content", "application_process"]:
# #                 content = data.get(key, [])
# #                 if isinstance(content, list):
# #                     text_parts.extend(content)

# #             chunk = "\n".join(text_parts).strip()
# #             if chunk:
# #                 chunks.append(chunk)
# #                 metadata.append({
# #                     "scheme_name": scheme_name,
# #                     "ministry": ministry,
# #                     "department": department
# #                 })

# #         return chunks, metadata

# #     def create_index(self):
# #         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks])

# #         if embeddings.ndim == 1:
# #             self.dimension = embeddings.shape[0]
# #             embeddings = embeddings.reshape(1, -1)
# #         else:
# #             self.dimension = embeddings.shape[1]

# #         self.index = faiss.IndexFlatL2(self.dimension)
# #         self.index.add(embeddings)

# #     def query(self, question, top_k=3):
# #         question_embedding = self.embedding_model.encode(question).reshape(1, -1)
# #         distances, indices = self.index.search(question_embedding, top_k)

# #         results = []
# #         for i in indices[0]:
# #             if i < len(self.chunks):
# #                 results.append({
# #                     "chunk": self.chunks[i],
# #                     "metadata": self.metadata[i]
# #                 })
# #         return results

# #     def generate_answer(self, question, context):
# #         hf_token = os.getenv("HUGGINGFACE_TOKEN")
# #         api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

# #         prompt = f"""
# # Context about government schemes:
# # {context}

# # Question: {question}

# # Given the context below about a government scheme, answer the user's question in 5 bullet points.

# # If available, mention:
# # - Scheme Name
# # - Purpose
# # - Eligibility
# # - Benefits
# # - Application Process
# # - Website Link

# # Highlight important section titles in **bold**. If any official website or source is found, attach it at the end.
# # If information is missing, simply skip that section.

# # Be concise but clear.
# # """

# #         headers = {
# #             "Authorization": f"Bearer {hf_token}",
# #             "Content-Type": "application/json"
# #         }
# #         payload = {
# #             "inputs": prompt,
# #             "options": {
# #                 "wait_for_model": True,
# #                 "max_length": 450,
# #                 "temperature": 0.1
# #             }
# #         }

# #         response = requests.post(api_url, headers=headers, json=payload)
# #         if response.status_code == 200:
# #             output = response.json()
# #             answer = output[0].get("generated_text", "No answer returned.")

# #             # Post-processing
# #             answer = answer.replace("Scheme Name:", "**Scheme Name:**")
# #             answer = answer.replace("Ministry/Department:", "**Ministry/Department:**")
# #             answer = answer.replace("Purpose:", "**Purpose:**")
# #             answer = answer.replace("Benefits:", "**Benefits:**")
# #             answer = answer.replace("Eligibility:", "**Eligibility:**")
# #             answer = answer.replace("Application Process:", "**Application Process:**")
# #             answer = answer.replace("Required Documents:", "**Required Documents:**")
# #             answer = answer.replace("Source:", "**Source:**")

# #             # Make URLs clickable
# #             if "**Source:**" in answer:
# #                 parts = answer.split("**Source:**")
# #                 if len(parts) > 1:
# #                     source_text = parts[1].strip()
# #                     urls = re.findall(r'https?://[^\s]+', source_text)
# #                     if urls:
# #                         for url in urls:
# #                             source_text = source_text.replace(url, f"[{url}]({url})")
# #                     parts[1] = source_text
# #                     answer = "**Source:**".join(parts)

# #             # Format application steps
# #             if "**Application Process:**" in answer:
# #                 parts = answer.split("**Application Process:**")
# #                 if len(parts) > 1:
# #                     step_text = parts[1]
# #                     for i in range(1, 10):
# #                         step_text = step_text.replace(f"{i}. ", f"\n{i}. ")
# #                     parts[1] = step_text
# #                     answer = "**Application Process:**".join(parts)

# #             return answer
# #         else:
# #             return f"Error: {response.status_code} - {response.text}"

#   
# # ## Version -2

# # import json
# # import numpy as np
# # import faiss
# # import os
# # import requests
# # from sentence_transformers import SentenceTransformer
# # from dotenv import load_dotenv
# # load_dotenv()



# # class GovernmentSchemeRAG:
# #     def __init__(self, json_path):
# #         self.json_path = json_path
# #         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# #         self.index = None
# #         self.dimension = None

# #         self.chunks, self.metadata = self.chunk_documents()
# #         if not self.chunks:
# #             raise ValueError("No chunks available to create embeddings.")

# #         self.create_index()

# #     def chunk_documents(self):
# #         chunks = []
# #         metadata = []

# #         with open(self.json_path, 'r') as f:
# #             self.schemes_data = json.load(f)

# #         for scheme in self.schemes_data:
# #             data = scheme.get("data", {})

# #             # Basic content
# #             text_parts = []
# #             scheme_name = data.get("scheme_name", "Unknown Scheme")
# #             ministry = data.get("ministry", "Unknown Ministry")
# #             department = data.get("department", "Unknown Department")

# #             text_parts.append(f"Scheme: {scheme_name}")
# #             text_parts.append(f"Ministry: {ministry}")
# #             text_parts.append(f"Department: {department}")

# #             # Add sections
# #             for key in ["details_content", "eligibility_content", "application_process"]:
# #                 content = data.get(key, [])
# #                 if isinstance(content, list):
# #                     text_parts.extend(content)

# #             chunk = "\n".join(text_parts).strip()
# #             if chunk:
# #                 chunks.append(chunk)
# #                 metadata.append({
# #                     "scheme_name": scheme_name,
# #                     "ministry": ministry,
# #                     "department": department
# #                 })

# #         return chunks, metadata

# #     def create_index(self):
# #         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks])

# #         if embeddings.ndim == 1:
# #             self.dimension = embeddings.shape[0]
# #             embeddings = embeddings.reshape(1, -1)
# #         else:
# #             self.dimension = embeddings.shape[1]

# #         self.index = faiss.IndexFlatL2(self.dimension)
# #         self.index.add(embeddings)

# #     def query(self, question, top_k=3):
# #         question_embedding = self.embedding_model.encode(question).reshape(1, -1)
# #         distances, indices = self.index.search(question_embedding, top_k)

# #         results = []
# #         for i in indices[0]:
# #             if i < len(self.chunks):
# #                 results.append({
# #                     "chunk": self.chunks[i],
# #                     "metadata": self.metadata[i]
# #                 })
# #         return results

#     # def generate_answer(self, question, context):
#     #     hf_token = os.getenv("HUGGINGFACE_TOKEN")
#     #     api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

#     #     prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
#     #     headers = {
#     #         "Authorization": f"Bearer {hf_token}",
#     #         "Content-Type": "application/json"
#     #     }
#     #     payload = {
#     #         "inputs": prompt,
#     #         "options": {"wait_for_model": True}
#     #     }

#     #     response = requests.post(api_url, headers=headers, json=payload)
#     #     if response.status_code == 200:
#     #         output = response.json()
#     #         return output[0].get("generated_text", "No answer returned.")
#     #     else:
#     #         return f"Error: {response.status_code} - {response.text}"


### Working code -1

# # import json
# # import numpy as np
# # import faiss
# # import os
# # import requests
# # from sentence_transformers import SentenceTransformer


# # class GovernmentSchemeRAG:
# #     def __init__(self, json_path):
# #         self.json_path = json_path
# #         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# #         self.index = None
# #         self.dimension = None

# #         self.chunks = self.chunk_documents()
# #         if not self.chunks:
# #             raise ValueError("No chunks available to create embeddings. Check your JSON data or chunk creation logic.")
        
# #         self.metadata = [{}] * len(self.chunks)  # placeholder if needed
# #         self.create_index()

# #     def chunk_documents(self):
# #         chunks = []
# #         with open(self.json_path, 'r') as f:
# #             self.schemes_data = json.load(f)  # ✅ Store here

# #         for scheme in self.schemes_data:
# #             data = scheme.get("data", {})

# #             text_parts = []
# #             if "scheme_name" in data:
# #                 text_parts.append(f"Scheme: {data['scheme_name']}")
# #             if "ministry" in data:
# #                 text_parts.append(f"Ministry: {data['ministry']}")
# #             if "department" in data:
# #                 text_parts.append(f"Department: {data['department']}")

# #             for key in ["details_content", "eligibility_content", "application_process"]:
# #                 content = data.get(key, [])
# #                 if isinstance(content, list):
# #                     text_parts.extend(content)

# #             chunk = "\n".join(text_parts).strip()
# #             if chunk:
# #                 chunks.append(chunk)

# #         return chunks

    
#     # def create_index(self):
#     #     # Create embeddings for all chunks
#     #     embeddings = self.embedding_model.encode(self.chunks)
        
#     #     # Convert to correct format for FAISS
#     #     embeddings = np.array(embeddings).astype('float32')
        
#     #     # Create FAISS index
#     #     self.dimension = embeddings.shape[1]
#     #     self.index = faiss.IndexFlatL2(self.dimension)
#     #     self.index.add(embeddings)
    
#     def create_index(self):
#     # Ensure chunks are not empty
#         if not self.chunks:
#             raise ValueError("No chunks available to create embeddings. Check your JSON data or chunk creation logic.")

#         # Generate embeddings for all chunks
#         embeddings = self.embedding_model.encode(self.chunks)

#         # Convert embeddings to a NumPy array
#         embeddings = np.array(embeddings).astype('float32')

#         # Ensure embeddings are 2D
#         if embeddings.ndim != 2 or embeddings.size == 0:
#             raise ValueError("Embeddings are not valid. Ensure the input data is correct and the embedding model is working.")

#         # Set the dimension for FAISS index
#         self.dimension = embeddings.shape[1]

#         # Create FAISS index
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.index.add(embeddings)
#         print(f"Total chunks created: {len(self.chunks)}")


#     def retrieve(self, query, top_k=3):
#         # Embed the query
#         query_embedding = self.embedding_model.encode([query])
#         query_embedding = np.array(query_embedding).astype('float32')
        
#         # Search the index
#         distances, indices = self.index.search(query_embedding, top_k)
        
#         # Get the retrieved chunks and their metadata
#         retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
#         retrieved_metadata = [self.metadata[idx] for idx in indices[0]]
        
#         return retrieved_chunks, retrieved_metadata, distances[0]
    
#     def generate_answer(self, query, retrieved_chunks):
#         # Combine retrieved chunks as context
#         context = "\n\n".join(retrieved_chunks)
        
#         # Create prompt
#         prompt = f"Answer the question based on the following information about government schemes:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        
#         # Tokenize and generate
#         inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
#         outputs = self.model.generate(
#             inputs.input_ids, 
#             max_length=512,
#             num_beams=4,
#             temperature=0.7,
#             early_stopping=True
#         )
        
#         answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return answer
    
#     def query(self, question, top_k=3):
#         # Retrieve relevant chunks
#         retrieved_chunks, retrieved_metadata, distances = self.retrieve(question, top_k)
        
#         # Generate answer
#         answer = self.generate_answer(question, retrieved_chunks)
        
#         return {
#             'answer': answer,
#             'sources': retrieved_metadata
#         }

