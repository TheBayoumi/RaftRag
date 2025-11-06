# RAFT-RAG Server

A friendly FastAPI server that lets you build your own AI assistant with **RAG (Retrieval Augmented Generation)** and **RAFT (Retrieval Augmented Fine-Tuning)** - all running locally on your machine!

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120.1-009688.svg)](https://fastapi.tiangolo.com/)

---

## 🎯 What Can You Do?

- **Ask questions about your documents** - Upload PDFs, text files, or Word docs and get AI-powered answers
- **Train your own AI model** - Fine-tune models on your data using RAFT
- **Everything runs locally** - No need for expensive API calls, your data stays private
- **Works with popular models** - Use models from HuggingFace like Llama, Mistral, and more

---

## 📦 What You Need

- **Python 3.11 or 3.12** (make sure it's installed!)
- **Windows, Mac, or Linux** (Windows users: WSL2 recommended)
- **GPU (optional)** - Makes things faster, but CPU works too!

---

## 🚀 Let's Get Started!

### Step 1: Get the Code

```bash
git clone https://github.com/TheBayoumi/RaftRag.git
cd RaftRag
```

### Step 2: Install Everything

**Option A: Automatic Installation (Recommended!)**

**Windows:**
```bash
install.bat
```

**Mac/Linux:**
```bash
chmod +x install.sh
./install.sh
```

These scripts will automatically:
- Create a virtual environment
- Install all dependencies in the correct order
- Verify everything is working

**Option B: Manual Installation (if you prefer manual setup)**

**⚠️ Important: Install in this exact order to avoid problems!**

```bash
# Step 1: Create a virtual environment
# Windows
python3.11 -m venv venv
venv\Scripts\activate

# Mac/Linux
python3.11 -m venv venv
source venv/bin/activate

# Step 2: Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Step 3: Install raganything (this installs most things you need)
pip install raganything==1.2.8

# Step 4: Install PyTorch with CUDA support (if you have an NVIDIA GPU)
pip install -r requirements-torch-cuda121.txt

# Step 5: Install everything else
pip install -r requirements.txt
```

### Step 3: Set Up Your Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and set at least these for RAG:
# - DEFAULT_BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2 (LLM for generating answers)
# - DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 (for document embeddings)
# - HF_TOKEN=your_token_here (if you're using models that need authentication)

# RAG Default Values (you can customize these):
# - similarity_threshold: 0.3 (how similar documents must be to your query)
# - top_k: 5 (number of document chunks to retrieve)
# - temperature: 0.1-0.2 (lower = more factual, higher = more creative)
```

#### Getting a Hugging Face Token

Some models (like Llama) need a token to download. Here's how to get one:

1. **Sign up** at [huggingface.co/join](https://huggingface.co/join)
2. **Create a token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Choose "Read" permission
   - Copy the token (you'll only see it once!)
3. **Add it to your `.env` file**:
   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
4. **Accept model licenses** - Visit the model page on HuggingFace and accept the license

**Note**: You don't need a token for public models like `mistralai/Mistral-7B-Instruct-v0.2`.

---

## 🏃 Quick Start Guide

### 1. Start the Server

```bash
python scripts/run_server.py
```

Or directly:
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Check if It's Running

Open your browser and visit:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

### 3. Upload Your Documents

First, upload a document to your collection:

```bash
curl -X POST "http://localhost:8000/api/v1/rag/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf" \
  -F "collection_name=my_documents"
```

Supported file types: PDF, TXT, MD, DOCX

### 4. Query Your Documents

Now you can ask questions about your uploaded documents. Here are three different ways to query:

#### Basic Query (Uses Default Settings)

The simplest way - just ask a question:

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "collection_name": "my_documents",
    "top_k": 5
  }'
```

#### Advanced Query (Custom Model & Settings)

Specify your own model and fine-tune the search parameters:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_name": "default_3",
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "query": "how to setup the company email on my mobile",
    "similarity_threshold": 0.3,
    "temperature": 0.2,
    "top_k": 5
  }'
```

**Parameters explained:**
- `model_name`: Which AI model to use for generating answers
- `similarity_threshold`: Minimum relevance score (0.3 = more results, 0.7 = fewer but better)
- `temperature`: Answer style (0.2 = factual, 0.7+ = creative)
- `top_k`: How many document chunks to retrieve (5 = balanced, 3 = focused)

#### Focused Query (Fewer, Higher Quality Results)

When you want fewer but more precise results, use a lower `top_k` with a higher `similarity_threshold`:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_name": "default_4",
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "query": "how to setup the company email on my mobile",
    "similarity_threshold": 0.7,
    "temperature": 0.2,
    "top_k": 3
  }'
```

**Why use this?** Returns 2-3 highly relevant sources instead of 5, perfect for focused answers with higher confidence scores.

### Understanding the Response

Every query returns a JSON response with:

- **`query`**: Your original question
- **`answer`**: The AI-generated answer based on your documents
- **`sources`**: Array of document chunks used, each with:
  - `content`: The relevant text excerpt
  - `score`: Confidence/relevance score (0-1)
  - `metadata`: File info (filename, doc_id, collection, etc.)
- **`num_sources`**: Total number of sources found
- **`processing_time`**: How long it took (in seconds)

#### Example Response 1: Focused Query (top_k: 3, similarity_threshold: 0.7)

```json
{
  "query": "how to setup the company email on my mobile",
  "answer": "To set up your company email on your mobile device, here are the step-by-step instructions as per the given text:\nSetting up your company email on your mobile device involves two main steps.\nFirstly, if your company requires Mobile Device Management (MDM), it is recommended to install the necessary profile on your device before proceeding.\nSecondly, once the MDM profile has been installed, you should be able to set up your company email account by following these simple steps:\nThe process typically begins by answering security-related questions during registration, such as verifying your identity through a one-time password sent via SMS or authenticating yourself against your existing credentials.\nAfter completing this initial verification, users may then proceed to create their new email address within the app settings of their mobile device.\nTherefore, setting up your company email on your mobile device follows these general guidelines outlined above.\n\n## Sources\n\n[1] rag_sample_qas_from_kis.csv\n[2] rag_sample_qas_from_kis.csv\n[3] Document 3",
  "sources": [
    {
      "content": "* If you need further assistance or have questions about company email policies, contact your IT department or refer to the company's email policy documentation., sample_question: \"How do I set up my company email on my mobile device?\", sample_ground_truth: To set up your company email on your mobile device, please follow these steps:",
      "score": 0.9838709677419354,
      "metadata": {
        "file_type": ".csv",
        "created_at": "2025-11-06T15:39:35.767102",
        "doc_id": "9e4ec0f5-7509-4905-9469-0252442c719e",
        "collection": "default_4",
        "database_type": "rag",
        "chunk_index": 7,
        "filename": "rag_sample_qas_from_kis.csv",
        "total_chunks": 118
      },
      "doc_id": "9e4ec0f5-7509-4905-9469-0252442c719e"
    },
    {
      "content": "**Step 1: Ensure Mobile Device Management (MDM) Profile is Installed (if required)**\n\nIf your company requires MDM for mobile devices, ensure that the profile is installed on your device. This profile will allow your device to connect to the company network and access company email. If you are unsure whether MDM is required, contact your IT department for assistance.\n\n**Step 2: Set Up Email Account on Mobile Device**",
      "score": 0.9242424242424242,
      "metadata": {
        "created_at": "2025-11-06T15:39:35.767102",
        "collection": "kaggle",
        "total_chunks": 118,
        "database_type": "rag",
        "doc_id": "9e4ec0f5-7509-4905-9469-0252442c719e",
        "chunk_index": 1,
        "file_type": ".csv",
        "filename": "rag_sample_qas_from_kis.csv"
      },
      "doc_id": "9e4ec0f5-7509-4905-9469-0252442c719e"
    }
  ],
  "num_sources": 2,
  "processing_time": 31.722424268722534
}
```

**Key observations:**
- Only 2 sources returned (high quality, focused results)
- High confidence scores (0.98 and 0.92)
- Fast processing time (~32 seconds)
- Answer includes step-by-step instructions with source citations

#### Example Response 2: Same Query, Different Collection (top_k: 3)

```json
{
  "query": "how to setup the company email on my mobile",
  "answer": "Here's the answer:\nTo set up your company email on your mobile device, follow these steps:\n1. Locate the Email app on your Android device.\nThe instructions provide step-by-step guidance on opening the Email app, but no specific details on adding a new account are given.\nHowever, we know that setting up email accounts typically involves creating one by selecting a provider such as Gmail, Outlook, Yahoo, etc.\nTherefore, considering this general process of setting up email accounts, here's what would likely happen when trying to add a company email account:\n* The user needs to select their preferred email service provider (e.g., Google Mail).\n* They may then enter their login credentials, including username and password.\n* After successful authentication, the system might prompt them to verify their identity through additional security measures if necessary.\nConsidering typical procedures for setting up corporate emails, users usually go through similar processes involving verifying identities before accessing work-related communications.\nHence, assuming standard practices, the correct sequence could involve initial verification followed by establishing communication channels within the organization.\nIn summary, while direct quotes from the original text don't explicitly state all the actions involved in setting up company email, common practices suggest that users must first create an account, log in, and potentially undergo some form of identification verification prior to receiving official correspondence related to workplace activities.\nSo, without more detailed explanations regarding each individual action taken during the setup procedure, our best educated guess at answering the query relies heavily upon understanding established protocols for employee email management systems commonly found across various organizations.\nReferences used:: Not directly mentioned; however, implied via general knowledge of business settings and online platforms like email services.: Sample ground truth explaining the process of setting up email accounts.: A reference providing basic instruction on how to locate and install the Email app on an Android device.\n\n## Sources\n\n[1] rag_sample_qas_from_kis.csv\n[2] rag_sample_qas_from_kis.csv\n[3] rag_sample_qas_from_kis.csv",
  "sources": [
    {
      "content": "* If you need further assistance or have questions about company email policies, contact your IT department or refer to the company's email policy documentation., sample_question: \"How do I set up my company email on my mobile device?\", sample_ground_truth: To set up your company email on your mobile device, please follow these steps:",
      "score": 1,
      "metadata": {
        "filename": "rag_sample_qas_from_kis.csv",
        "doc_id": "9dd8acb0-3eed-4416-9bc9-5f1ab617ce53",
        "chunk_index": 7,
        "file_type": ".csv",
        "total_chunks": 118,
        "created_at": "2025-11-06T15:39:55.778266",
        "collection": "kaggle",
        "database_type": "rag"
      },
      "doc_id": "9dd8acb0-3eed-4416-9bc9-5f1ab617ce53"
    },
    {
      "content": "**Step 1: Ensure Mobile Device Management (MDM) Profile is Installed (if required)**\n\nIf your company requires MDM for mobile devices, ensure that the profile is installed on your device. This profile will allow your device to connect to the company network and access company email. If you are unsure whether MDM is required, contact your IT department for assistance.\n\n**Step 2: Set Up Email Account on Mobile Device**",
      "score": 0.9682539682539681,
      "metadata": {
        "doc_id": "9dd8acb0-3eed-4416-9bc9-5f1ab617ce53",
        "file_type": ".csv",
        "filename": "rag_sample_qas_from_kis.csv",
        "created_at": "2025-11-06T15:39:55.778266",
        "chunk_index": 1,
        "collection": "kaggle",
        "total_chunks": 118,
        "database_type": "rag"
      },
      "doc_id": "9dd8acb0-3eed-4416-9bc9-5f1ab617ce53"
    },
    {
      "content": "By following these steps, you should be able to successfully configure email on your Android device., sample_question: How do I set up my company email on my personal Android device?, sample_ground_truth: To set up your company email on your personal Android device, follow these steps:\n\n**Step 1: Open the Email App**\n\nLocate the Email app on your Android device and tap on it to open it.\n\n**Step 2: Add a New Account**",
      "score": 0.9540566959921798,
      "metadata": {
        "database_type": "rag",
        "total_chunks": 118,
        "collection": "kaggle",
        "created_at": "2025-11-06T15:39:55.778266",
        "filename": "rag_sample_qas_from_kis.csv",
        "doc_id": "9dd8acb0-3eed-4416-9bc9-5f1ab617ce53",
        "chunk_index": 113,
        "file_type": ".csv"
      },
      "doc_id": "9dd8acb0-3eed-4416-9bc9-5f1ab617ce53"
    }
  ],
  "num_sources": 3,
  "processing_time": 64.84165740013123
}
```

**Key observations:**
- 3 sources returned (as requested with top_k: 3)
- Perfect score (1.0) on the most relevant source
- All sources from the same document with high relevance
- Detailed answer with step-by-step instructions

---

## 📚 Common Tasks

### Upload Documents
```
POST /api/v1/rag/documents
Content-Type: multipart/form-data

Form data:
- file: Your file (PDF, TXT, MD, DOCX)
- collection_name: Name for your document collection
```

### Query Documents
```
POST /api/v1/rag/query
Content-Type: application/json

{
  "query": "Your question here",
  "collection_name": "your_collection",
  "top_k": 5
}
```

### Query with Custom Model
```
POST /api/v1/query
Content-Type: application/json

{
  "collection_name": "your_collection",
  "model_name": "meta-llama/Llama-3.2-1B-Instruct",
  "query": "Your question here",
  "similarity_threshold": 0.3,
  "temperature": 0.2,
  "top_k": 5
}
```

---

## ⚙️ Basic Configuration

Edit your `.env` file to customize:

```env
# Model Settings
DEFAULT_BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# HuggingFace Token (for gated models)
HF_TOKEN=your_token_here

# Server Settings
HOST=0.0.0.0
PORT=8000
```

See `.env.example` for all options.

---

## 💡 Tips

- **Start simple**: Upload a text file and ask basic questions first
- **Use the API docs**: Visit `http://localhost:8000/docs` to try endpoints interactively
- **Check the logs**: If something goes wrong, check the console output
- **GPU not required**: Everything works on CPU, just slower
- **Small models first**: Try `Llama-3.2-1B-Instruct` before larger models
- **Adjust top_k**: Lower values (3) give fewer but more relevant results; higher values (5-10) give more context

---

## 🛠️ Need Help?

- **Check the API docs**: http://localhost:8000/docs (when server is running)
- **Open an issue**: [GitHub Issues](https://github.com/TheBayoumi/RaftRag/issues)
- **Read the code**: The codebase is well-documented with type hints

---

## 📄 License

MIT License - feel free to use this for your projects!

---

## 👤 Author

**Mahmoud BAYOUMI**

- Email: mahmoud.bayoumi.pp@gmail.com
- GitHub: [@TheBayoumi](https://github.com/TheBayoumi)

---

**Happy coding! 🚀**
