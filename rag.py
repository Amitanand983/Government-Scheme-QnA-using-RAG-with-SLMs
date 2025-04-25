### APi key input 
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import re

class GovernmentSchemeRAG:
    def __init__(self, json_path, hf_token=""):
        self.json_path = json_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.dimension = None

        # API Key provided via parameter (from Streamlit input)
        self.hf_token = hf_token

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
Context about government schemes:
{context}

Question: {question}

Given the context below about a government scheme, answer the user's question concisely, focusing on the key details requested.

If available, mention:
- Scheme Name
- Purpose
- Eligibility
- Key Benefits
- Application Process Overview (briefly)
- Website Link (if explicitly found in context)

Highlight important section titles in **bold**.
If information is missing for a section, simply omit that section. Be clear and direct.
"""
        answer = "Could not generate answer using Hugging Face."  # Default error message

        if self.hf_token:
            api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
            headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
            payload = {"inputs": prompt, "options": {"wait_for_model": True, "max_length": 450, "temperature": 0.1}}
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                output = response.json()
                if output and isinstance(output, list) and 'generated_text' in output[0]:
                    answer = output[0].get("generated_text", "No answer returned by Flan-T5.")
                else:
                    answer = f"Unexpected response format from Flan-T5 API: {output}"
            except requests.exceptions.RequestException as e:
                print(f"Error calling Hugging Face API: {e}")
                answer = f"Error: Could not connect to Hugging Face API - {e}"
            except Exception as e:
                print(f"Error processing Hugging Face response: {e}")
                answer = f"Error processing Hugging Face response: {e}"
        else:
            answer = "Hugging Face model unavailable (check HUGGINGFACE_TOKEN input)."

        # Apply post-processing
        answer = answer.replace("Scheme Name:", "**Scheme Name:**")
        answer = answer.replace("Ministry/Department:", "**Ministry/Department:**")
        answer = answer.replace("Purpose:", "**Purpose:**")
        answer = answer.replace("Benefits:", "**Benefits:**")
        answer = answer.replace("Key Benefits:", "**Key Benefits:**")
        answer = answer.replace("Eligibility:", "**Eligibility:**")
        answer = answer.replace("Application Process:", "**Application Process:**")
        answer = answer.replace("Application Process Overview:", "**Application Process Overview:**")
        answer = answer.replace("Required Documents:", "**Required Documents:**")
        answer = answer.replace("Website Link:", "**Website Link:**")
        answer = answer.replace("Source:", "**Source:**")  # Keep this if your prompt might generate it

        # Make URLs clickable
        urls = re.findall(r'(https?://[^\s]+)', answer)
        for url in urls:
            # Basic check to avoid mangling markdown links if already formatted
            if f"[{url}]({url})" not in answer and f"**Website Link:** {url}" in answer:
                answer = answer.replace(url, f"[{url}]({url})")

        # Format application steps (basic newline formatting)
        if "**Application Process:**" in answer or "**Application Process Overview:**" in answer:
            lines = answer.split('\n')
            formatted_lines = []
            in_app_process = False
            for line in lines:
                if line.strip().startswith("**Application Process"):
                    in_app_process = True
                    formatted_lines.append(line)
                elif in_app_process and re.match(r'^\s*\d+\.\s+', line.strip()):
                    formatted_lines.append(line.strip())  # Keep numbered steps
                elif in_app_process and line.strip().startswith('- '):
                    formatted_lines.append(line.strip())  # Keep bullet points
                elif in_app_process and line.strip() == "":
                    # Stop adding newlines if the section seems to end
                    if len(formatted_lines) > 0 and formatted_lines[-1].strip() != "":
                        in_app_process = False  # Assume end of section on blank line
                    formatted_lines.append(line)  # Keep blank lines within reason
                elif in_app_process:
                    formatted_lines.append(line)  # Keep other lines in the section
                else:
                    formatted_lines.append(line)  # Add lines outside the section
            answer = "\n".join(formatted_lines)

        return answer


# # # rag.py

# import json
# import numpy as np
# import faiss
# import os
# import requests
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import re
# import google.generativeai as genai # Added for Gemini

# load_dotenv()

# class GovernmentSchemeRAG:
#     def __init__(self, json_path):
#         self.json_path = json_path
#         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.index = None
#         self.dimension = None

#         # Load API Keys
#         self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
#         self.google_api_key = os.getenv("GOOGLE_API_KEY")

#         # Configure Gemini
#         if self.google_api_key:
#             try:
#                 genai.configure(api_key=self.google_api_key)
#             except Exception as e:
#                 print(f"Error configuring Gemini API: {e}") # Non-blocking error
#                 self.google_api_key = None # Disable Gemini if config fails
#         else:
#              print("Warning: GOOGLE_API_KEY not found in .env file. Gemini model will be unavailable.")


#         self.chunks, self.metadata = self.chunk_documents()
#         if not self.chunks:
#             raise ValueError("No chunks available to create embeddings.")

#         self.create_index()

#     def chunk_documents(self):
#         chunks = []
#         metadata = []
#         try:
#             with open(self.json_path, 'r', encoding='utf-8') as f: # Specify encoding
#                 self.schemes_data = json.load(f)
#         except FileNotFoundError:
#             st.error(f"Error: JSON file not found at {self.json_path}")
#             return [], []
#         except json.JSONDecodeError:
#             st.error(f"Error: Could not decode JSON from {self.json_path}. Check file format.")
#             return [], []
#         except Exception as e:
#             st.error(f"An unexpected error occurred loading the JSON: {e}")
#             return [], []


#         for scheme in self.schemes_data:
#             data = scheme.get("data", {})

#             text_parts = []
#             scheme_name = data.get("scheme_name", "Unknown Scheme")
#             ministry = data.get("ministry", "Unknown Ministry")
#             department = data.get("department", "Unknown Department")

#             text_parts.append(f"Scheme: {scheme_name}")
#             text_parts.append(f"Ministry: {ministry}")
#             text_parts.append(f"Department: {department}")

#             for key in ["details_content", "eligibility_content", "application_process"]:
#                 content = data.get(key, [])
#                 if isinstance(content, list):
#                      # Clean up potential None values or non-string items if necessary
#                     cleaned_content = [str(item) for item in content if item is not None]
#                     text_parts.extend(cleaned_content)
#                 elif content is not None: # Handle cases where it might be a single string
#                     text_parts.append(str(content))


#             chunk = "\n".join(text_parts).strip()
#             if chunk:
#                 chunks.append(chunk)
#                 metadata.append({
#                     "scheme_name": scheme_name,
#                     "ministry": ministry,
#                     "department": department
#                 })

#         return chunks, metadata

#     def create_index(self):
#         if not self.chunks:
#             print("Skipping index creation as no chunks were loaded.")
#             return
#         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks]).astype('float32') # Ensure float32

#         if embeddings.ndim == 1:
#              if embeddings.shape[0] > 0: # Check if the single dimension is not empty
#                 self.dimension = embeddings.shape[0]
#                 embeddings = embeddings.reshape(1, -1)
#              else:
#                  print("Warning: Embeddings array is empty or invalid.")
#                  return # Cannot create index with empty embeddings
#         elif embeddings.shape[0] == 0: # Check if the 2D array has no rows
#             print("Warning: Embeddings array is empty.")
#             return # Cannot create index with empty embeddings
#         else:
#             self.dimension = embeddings.shape[1]


#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.index.add(embeddings)
#         print(f"FAISS index created successfully with {self.index.ntotal} vectors.")


#     def query(self, question, top_k=3):
#         if not self.index or self.index.ntotal == 0:
#              return [] # Return empty if index doesn't exist or is empty
#         question_embedding = self.embedding_model.encode(question).reshape(1, -1).astype('float32') # Ensure float32
#         distances, indices = self.index.search(question_embedding, top_k)

#         results = []
#         for i in indices[0]:
#             # Check index bounds robustly
#             if 0 <= i < len(self.chunks):
#                 results.append({
#                     "chunk": self.chunks[i],
#                     "metadata": self.metadata[i]
#                 })
#             else:
#                  print(f"Warning: Index {i} out of bounds for chunks list (length {len(self.chunks)}).")
#         return results

#     # --- Updated generate_answer ---
#     def generate_answer(self, question, context, model_choice='HuggingFace'):
#         prompt = f"""
# Context about government schemes:
# {context}

# Question: {question}

# Given the context below about a government scheme, answer the user's question concisely, focusing on the key details requested.

# If available, mention:
# - Scheme Name
# - Purpose
# - Eligibility
# - Key Benefits
# - Application Process Overview (briefly)
# - Website Link (if explicitly found in context)

# Highlight important section titles in **bold**.
# If information is missing for a section, simply omit that section. Be clear and direct.
# """
#         answer = f"Could not generate answer using {model_choice}." # Default error message

#         if model_choice == 'Gemini' and self.google_api_key:
#             try:
#                 generation_config = {"temperature": 0.1, "max_output_tokens": 500} # Simplified config
#                 safety_settings=[ # Basic safety settings
#                     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#                     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#                     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#                     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#                 ]
#                 model = genai.GenerativeModel(model_name="gemini-1.0-pro",
#                                               generation_config=generation_config,
#                                               safety_settings=safety_settings)
#                 response = model.generate_content(prompt)
#                 answer = response.text
#             except Exception as e:
#                 print(f"Error generating answer with Gemini: {e}")
#                 answer = f"Error generating answer with Gemini: {e}"

#         elif model_choice == 'HuggingFace' and self.hf_token:
#             api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
#             headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
#             payload = {"inputs": prompt, "options": {"wait_for_model": True, "max_length": 450, "temperature": 0.1}}
#             try:
#                 response = requests.post(api_url, headers=headers, json=payload)
#                 response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
#                 output = response.json()
#                 if output and isinstance(output, list) and 'generated_text' in output[0]:
#                      answer = output[0].get("generated_text", "No answer returned by Flan-T5.")
#                 else:
#                      answer = f"Unexpected response format from Flan-T5 API: {output}"

#             except requests.exceptions.RequestException as e:
#                 print(f"Error calling Hugging Face API: {e}")
#                 answer = f"Error: Could not connect to Hugging Face API - {e}"
#             except Exception as e:
#                 print(f"Error processing Hugging Face response: {e}")
#                 answer = f"Error processing Hugging Face response: {e}"

#         else:
#              if model_choice == 'Gemini':
#                  answer = "Gemini model unavailable (check GOOGLE_API_KEY in .env)."
#              elif model_choice == 'HuggingFace':
#                  answer = "Hugging Face model unavailable (check HUGGINGFACE_TOKEN in .env)."


#         # Apply post-processing (Common for both models)
#         answer = answer.replace("Scheme Name:", "**Scheme Name:**")
#         answer = answer.replace("Ministry/Department:", "**Ministry/Department:**")
#         answer = answer.replace("Purpose:", "**Purpose:**")
#         answer = answer.replace("Benefits:", "**Benefits:**")
#         answer = answer.replace("Key Benefits:", "**Key Benefits:**")
#         answer = answer.replace("Eligibility:", "**Eligibility:**")
#         answer = answer.replace("Application Process:", "**Application Process:**")
#         answer = answer.replace("Application Process Overview:", "**Application Process Overview:**")
#         answer = answer.replace("Required Documents:", "**Required Documents:**")
#         answer = answer.replace("Website Link:", "**Website Link:**")
#         answer = answer.replace("Source:", "**Source:**") # Keep this if your prompt might generate it

#         # Make URLs clickable
#         urls = re.findall(r'(https?://[^\s]+)', answer)
#         for url in urls:
#              # Basic check to avoid mangling markdown links if already formatted
#              if f"[{url}]({url})" not in answer and f"**Website Link:** {url}" in answer :
#                  answer = answer.replace(url, f"[{url}]({url})")

#         # Format application steps (basic newline formatting)
#         if "**Application Process:**" in answer or "**Application Process Overview:**" in answer:
#             lines = answer.split('\n')
#             formatted_lines = []
#             in_app_process = False
#             for line in lines:
#                 if line.strip().startswith("**Application Process"):
#                      in_app_process = True
#                      formatted_lines.append(line)
#                 elif in_app_process and re.match(r'^\s*\d+\.\s+', line.strip()):
#                      formatted_lines.append(line.strip()) # Keep numbered steps
#                 elif in_app_process and line.strip().startswith('- '):
#                      formatted_lines.append(line.strip()) # Keep bullet points
#                 elif in_app_process and line.strip() == "":
#                     # Stop adding newlines if the section seems to end
#                     if len(formatted_lines) > 0 and formatted_lines[-1].strip() != "":
#                         in_app_process = False # Assume end of section on blank line
#                     formatted_lines.append(line) # Keep blank lines within reason
#                 elif in_app_process:
#                      formatted_lines.append(line) # Keep other lines in the section
#                 else:
#                      formatted_lines.append(line) # Add lines outside the section
#             answer = "\n".join(formatted_lines)


#         return answer

# ## Version -3 compatible with version -3 of main.py

# import json
# import numpy as np
# import faiss
# import os
# import requests
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import re

# load_dotenv()

# class GovernmentSchemeRAG:
#     def __init__(self, json_path):
#         self.json_path = json_path
#         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.index = None
#         self.dimension = None

#         self.chunks, self.metadata = self.chunk_documents()
#         if not self.chunks:
#             raise ValueError("No chunks available to create embeddings.")

#         self.create_index()

#     def chunk_documents(self):
#         chunks = []
#         metadata = []

#         with open(self.json_path, 'r') as f:
#             self.schemes_data = json.load(f)

#         for scheme in self.schemes_data:
#             data = scheme.get("data", {})

#             text_parts = []
#             scheme_name = data.get("scheme_name", "Unknown Scheme")
#             ministry = data.get("ministry", "Unknown Ministry")
#             department = data.get("department", "Unknown Department")

#             text_parts.append(f"Scheme: {scheme_name}")
#             text_parts.append(f"Ministry: {ministry}")
#             text_parts.append(f"Department: {department}")

#             for key in ["details_content", "eligibility_content", "application_process"]:
#                 content = data.get(key, [])
#                 if isinstance(content, list):
#                     text_parts.extend(content)

#             chunk = "\n".join(text_parts).strip()
#             if chunk:
#                 chunks.append(chunk)
#                 metadata.append({
#                     "scheme_name": scheme_name,
#                     "ministry": ministry,
#                     "department": department
#                 })

#         return chunks, metadata

#     def create_index(self):
#         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks])

#         if embeddings.ndim == 1:
#             self.dimension = embeddings.shape[0]
#             embeddings = embeddings.reshape(1, -1)
#         else:
#             self.dimension = embeddings.shape[1]

#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.index.add(embeddings)

#     def query(self, question, top_k=3):
#         question_embedding = self.embedding_model.encode(question).reshape(1, -1)
#         distances, indices = self.index.search(question_embedding, top_k)

#         results = []
#         for i in indices[0]:
#             if i < len(self.chunks):
#                 results.append({
#                     "chunk": self.chunks[i],
#                     "metadata": self.metadata[i]
#                 })
#         return results

#     def generate_answer(self, question, context):
#         hf_token = os.getenv("HUGGINGFACE_TOKEN")
#         api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"

#         prompt = f"""
# Context about government schemes:
# {context}

# Question: {question}

# Given the context below about a government scheme, answer the user's question in 5 bullet points.

# If available, mention:
# - Scheme Name
# - Purpose
# - Eligibility
# - Benefits
# - Application Process
# - Website Link

# Highlight important section titles in **bold**. If any official website or source is found, attach it at the end.
# If information is missing, simply skip that section.

# Be concise but clear.
# """

#         headers = {
#             "Authorization": f"Bearer {hf_token}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "inputs": prompt,
#             "options": {
#                 "wait_for_model": True,
#                 "max_length": 450,
#                 "temperature": 0.1
#             }
#         }

#         response = requests.post(api_url, headers=headers, json=payload)
#         if response.status_code == 200:
#             output = response.json()
#             answer = output[0].get("generated_text", "No answer returned.")

#             # Post-processing
#             answer = answer.replace("Scheme Name:", "**Scheme Name:**")
#             answer = answer.replace("Ministry/Department:", "**Ministry/Department:**")
#             answer = answer.replace("Purpose:", "**Purpose:**")
#             answer = answer.replace("Benefits:", "**Benefits:**")
#             answer = answer.replace("Eligibility:", "**Eligibility:**")
#             answer = answer.replace("Application Process:", "**Application Process:**")
#             answer = answer.replace("Required Documents:", "**Required Documents:**")
#             answer = answer.replace("Source:", "**Source:**")

#             # Make URLs clickable
#             if "**Source:**" in answer:
#                 parts = answer.split("**Source:**")
#                 if len(parts) > 1:
#                     source_text = parts[1].strip()
#                     urls = re.findall(r'https?://[^\s]+', source_text)
#                     if urls:
#                         for url in urls:
#                             source_text = source_text.replace(url, f"[{url}]({url})")
#                     parts[1] = source_text
#                     answer = "**Source:**".join(parts)

#             # Format application steps
#             if "**Application Process:**" in answer:
#                 parts = answer.split("**Application Process:**")
#                 if len(parts) > 1:
#                     step_text = parts[1]
#                     for i in range(1, 10):
#                         step_text = step_text.replace(f"{i}. ", f"\n{i}. ")
#                     parts[1] = step_text
#                     answer = "**Application Process:**".join(parts)

#             return answer
#         else:
#             return f"Error: {response.status_code} - {response.text}"

  
# # ## Version -2

# import json
# import numpy as np
# import faiss
# import os
# import requests
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# load_dotenv()



# class GovernmentSchemeRAG:
#     def __init__(self, json_path):
#         self.json_path = json_path
#         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.index = None
#         self.dimension = None

#         self.chunks, self.metadata = self.chunk_documents()
#         if not self.chunks:
#             raise ValueError("No chunks available to create embeddings.")

#         self.create_index()

#     def chunk_documents(self):
#         chunks = []
#         metadata = []

#         with open(self.json_path, 'r') as f:
#             self.schemes_data = json.load(f)

#         for scheme in self.schemes_data:
#             data = scheme.get("data", {})

#             # Basic content
#             text_parts = []
#             scheme_name = data.get("scheme_name", "Unknown Scheme")
#             ministry = data.get("ministry", "Unknown Ministry")
#             department = data.get("department", "Unknown Department")

#             text_parts.append(f"Scheme: {scheme_name}")
#             text_parts.append(f"Ministry: {ministry}")
#             text_parts.append(f"Department: {department}")

#             # Add sections
#             for key in ["details_content", "eligibility_content", "application_process"]:
#                 content = data.get(key, [])
#                 if isinstance(content, list):
#                     text_parts.extend(content)

#             chunk = "\n".join(text_parts).strip()
#             if chunk:
#                 chunks.append(chunk)
#                 metadata.append({
#                     "scheme_name": scheme_name,
#                     "ministry": ministry,
#                     "department": department
#                 })

#         return chunks, metadata

#     def create_index(self):
#         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks])

#         if embeddings.ndim == 1:
#             self.dimension = embeddings.shape[0]
#             embeddings = embeddings.reshape(1, -1)
#         else:
#             self.dimension = embeddings.shape[1]

#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.index.add(embeddings)

#     def query(self, question, top_k=3):
#         question_embedding = self.embedding_model.encode(question).reshape(1, -1)
#         distances, indices = self.index.search(question_embedding, top_k)

#         results = []
#         for i in indices[0]:
#             if i < len(self.chunks):
#                 results.append({
#                     "chunk": self.chunks[i],
#                     "metadata": self.metadata[i]
#                 })
#         return results

    # def generate_answer(self, question, context):
    #     hf_token = os.getenv("HUGGINGFACE_TOKEN")
    #     api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"

    #     prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    #     headers = {
    #         "Authorization": f"Bearer {hf_token}",
    #         "Content-Type": "application/json"
    #     }
    #     payload = {
    #         "inputs": prompt,
    #         "options": {"wait_for_model": True}
    #     }

    #     response = requests.post(api_url, headers=headers, json=payload)
    #     if response.status_code == 200:
    #         output = response.json()
    #         return output[0].get("generated_text", "No answer returned.")
    #     else:
    #         return f"Error: {response.status_code} - {response.text}"


### Working code -1

# import json
# import numpy as np
# import faiss
# import os
# import requests
# from sentence_transformers import SentenceTransformer


# class GovernmentSchemeRAG:
#     def __init__(self, json_path):
#         self.json_path = json_path
#         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.index = None
#         self.dimension = None

#         self.chunks = self.chunk_documents()
#         if not self.chunks:
#             raise ValueError("No chunks available to create embeddings. Check your JSON data or chunk creation logic.")
        
#         self.metadata = [{}] * len(self.chunks)  # placeholder if needed
#         self.create_index()

#     def chunk_documents(self):
#         chunks = []
#         with open(self.json_path, 'r') as f:
#             self.schemes_data = json.load(f)  # ✅ Store here

#         for scheme in self.schemes_data:
#             data = scheme.get("data", {})

#             text_parts = []
#             if "scheme_name" in data:
#                 text_parts.append(f"Scheme: {data['scheme_name']}")
#             if "ministry" in data:
#                 text_parts.append(f"Ministry: {data['ministry']}")
#             if "department" in data:
#                 text_parts.append(f"Department: {data['department']}")

#             for key in ["details_content", "eligibility_content", "application_process"]:
#                 content = data.get(key, [])
#                 if isinstance(content, list):
#                     text_parts.extend(content)

#             chunk = "\n".join(text_parts).strip()
#             if chunk:
#                 chunks.append(chunk)

#         return chunks

#     def create_index(self):
#         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks])

#         # Handle 1D (single vector) or 2D cases
#         if embeddings.ndim == 1:
#             self.dimension = embeddings.shape[0]
#             embeddings = embeddings.reshape(1, -1)
#         else:
#             self.dimension = embeddings.shape[1]

#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.index.add(embeddings)

#     def query(self, question, top_k=3):
#         question_embedding = self.embedding_model.encode(question).reshape(1, -1)
#         distances, indices = self.index.search(question_embedding, top_k)

#         results = []
#         for i in indices[0]:
#             if i < len(self.chunks):
#                 results.append(self.chunks[i])
#         return results

#     def generate_answer(self, question, context):
#         hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Or hardcode your token here
#         api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"

#         prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
#         headers = {
#             "Authorization": f"Bearer {hf_token}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "inputs": prompt,
#             "options": {"wait_for_model": True}
#         }

#         response = requests.post(api_url, headers=headers, json=payload)
#         if response.status_code == 200:
#             output = response.json()
#             return output[0].get("generated_text", "No answer returned.")
#         else:
#             return f"Error: {response.status_code} - {response.text}"


# # rag.py - Core RAG functionality
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# import faiss
# import torch
# import streamlit as st

# class GovernmentSchemeRAG:
#     def __init__(self, json_path):
#         # Initialize embedding model
#         self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
#         # Initialize LLM
#         self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
#         self.model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
        
#         # Load and process data
#         if json_path:
#             self.schemes_data = self.load_data(json_path)
#             self.chunks, self.metadata = self.create_chunks()
            
#             # Create FAISS index
#             self.create_index()
    
#     # def load_data(self, json_path):
#     #     try:
#     #         with open(json_path, 'r', encoding='utf-8') as f:
#     #             data = json.load(f)
#     #         print(f"Successfully loaded JSON with {len(data)} records")
#     #         return data
#     #     except json.JSONDecodeError as e:
#     #         print(f"Error parsing JSON: {e}")
#     def load_data(self, json_path):
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             print(f"Successfully loaded JSON with {len(data)} records")
#             print(f"Sample data: {data[:2]}")  # Print a sample of the data
#             return data
#         except json.JSONDecodeError as e:
#             print(f"Error parsing JSON: {e}")
            
#             # Try to load the file line by line to handle partial data
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
            
#             # If the file isn't line-delimited JSON, try to parse as much as possible
#             try:
#                 # Find the last valid JSON array
#                 valid_json = "[" + ",".join([line.strip() for line in lines if line.strip() and line.strip() not in [',', '[', ']']]) + "]"
#                 data = json.loads(valid_json)
#                 print(f"Loaded {len(data)} items from partially valid JSON")
#                 return data
#             except json.JSONDecodeError:
#                 # If that fails, try to load each line as a separate JSON object
#                 valid_records = []
#                 for i, line in enumerate(lines):
#                     try:
#                         if line.strip() and not line.strip() in [',', '[', ']']:
#                             # Remove trailing commas that might cause problems
#                             clean_line = line.strip().rstrip(',')
#                             if clean_line:
#                                 # Check if it's a standalone object or part of an array
#                                 if clean_line.startswith('{'):
#                                     record = json.loads(clean_line)
#                                     valid_records.append(record)
#                     except json.JSONDecodeError:
#                         continue
                    
#                     # Print progress occasionally
#                     if i % 10000 == 0:
#                         print(f"Processed {i} lines, found {len(valid_records)} valid records")
                
#                 if valid_records:
#                     print(f"Loaded {len(valid_records)} items from line-by-line parsing")
#                     return valid_records
#                 else:
#                     print("Could not parse any valid JSON records")
                    
#                     # Last resort: try to fix the JSON file
#                     print("Attempting to fix JSON file...")
#                     fixed_data = self.fix_json_file(json_path)
#                     return fixed_data if fixed_data else []

#     def fix_json_file(self, json_path):
#         """Attempt to fix a corrupted JSON file by parsing it as text and reconstructing."""
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
            
#             # Check if it's an array
#             content = content.strip()
#             if not (content.startswith('[') and content.endswith(']')):
#                 print("JSON does not appear to be an array")
#                 return []
            
#             # Remove the outer brackets
#             content = content[1:-1].strip()
            
#             # Split by what appears to be object boundaries
#             objects = []
#             object_text = ""
#             brace_count = 0
#             in_string = False
#             escape_next = False
            
#             for char in content:
#                 if escape_next:
#                     escape_next = False
#                     object_text += char
#                     continue
                    
#                 if char == '\\':
#                     escape_next = True
#                     object_text += char
#                     continue
                    
#                 if char == '"' and not escape_next:
#                     in_string = not in_string
#                     object_text += char
#                     continue
                    
#                 if not in_string:
#                     if char == '{':
#                         brace_count += 1
#                         if brace_count == 1:
#                             object_text = '{'
#                             continue
#                     elif char == '}':
#                         brace_count -= 1
#                         if brace_count == 0:
#                             object_text += '}'
#                             try:
#                                 obj = json.loads(object_text)
#                                 objects.append(obj)
#                                 if len(objects) % 100 == 0:
#                                     print(f"Found valid object, total: {len(objects)}")
#                             except json.JSONDecodeError:
#                                 pass
#                             object_text = ""
#                             continue
                
#                 if brace_count > 0:
#                     object_text += char
            
#             print(f"Fixed JSON parsing found {len(objects)} valid objects")
#             return objects
#         except Exception as e:
#             print(f"Error attempting to fix JSON: {e}")
#             return []
    
#     # def create_chunks(self):
#     #     chunks = []
#     #     metadata = []
        
#     #     for page in self.schemes_data:
#     #         scheme_data = page.get('data', {})
            
#     #         # Extract scheme information
#     #         scheme_name = scheme_data.get('scheme_name', '')
#     #         ministry = scheme_data.get('ministry', '')
#     #         department = scheme_data.get('department', '')
            
#     #         # Process details content
#     #         details = ' '.join(scheme_data.get('details_content', []))
            
#     #         # Process eligibility content
#     #         eligibility = ' '.join(scheme_data.get('eligibility_content', []))
            
#     #         # Process application process
#     #         application = ' '.join(scheme_data.get('application_process', []))
            
#     #         # Process tags
#     #         tags = ', '.join(scheme_data.get('tags', []))
            
#     #         # Create main chunk with all information
#     #         main_chunk = f"Scheme: {scheme_name}\nMinistry: {ministry}\nDepartment: {department}\n"
#     #         main_chunk += f"Details: {details}\nEligibility: {eligibility}\n"
#     #         main_chunk += f"Application Process: {application}\nTags: {tags}"
            
#     #         chunks.append(main_chunk)
#     #         metadata.append({
#     #             'scheme_name': scheme_name,
#     #             'ministry': ministry,
#     #             'department': department,
#     #             'page': page.get('page', 0)
#     #         })
            
#     #         # Create additional smaller chunks for specific aspects
#     #         if details:
#     #             chunks.append(f"Scheme: {scheme_name}\nDetails: {details}")
#     #             metadata.append({
#     #                 'scheme_name': scheme_name,
#     #                 'type': 'details',
#     #                 'page': page.get('page', 0)
#     #             })
            
#     #         if eligibility:
#     #             chunks.append(f"Scheme: {scheme_name}\nEligibility: {eligibility}")
#     #             metadata.append({
#     #                 'scheme_name': scheme_name,
#     #                 'type': 'eligibility',
#     #                 'page': page.get('page', 0)
#     #             })
            
#     #         if application:
#     #             chunks.append(f"Scheme: {scheme_name}\nApplication Process: {application}")
#     #             metadata.append({
#     #                 'scheme_name': scheme_name,
#     #                 'type': 'application',
#     #                 'page': page.get('page', 0)
#     #             })
        
#     #     return chunks, metadata

#     def chunk_documents(self):
#         chunks = []

#         with open(self.json_path, 'r') as f:
#             schemes = json.load(f)

#         for scheme in schemes:
#             data = scheme.get("data", {})

#             text_parts = []

#             # Basic info
#             text_parts.append(f"Scheme: {data.get('scheme_name', '')}")
#             text_parts.append(f"Ministry: {data.get('ministry', '')}")
#             text_parts.append(f"Department: {data.get('department', '')}")

#             # Add list-based content if available
#             for key in ["details_content", "eligibility_content", "application_process"]:
#                 if key in data and isinstance(data[key], list):
#                     text_parts.extend(data[key])

#             # Join all parts into one chunk
#             chunk = "\n".join(text_parts).strip()
#             if chunk:
#                 chunks.append(chunk)

#         return chunks

    
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

