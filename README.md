# 🍕 RAG-Based AI Agent for Restaurant Review Question Answering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange.svg)](https://ollama.ai/)

An intelligent AI agent built with **RAG (Retrieval-Augmented Generation)** that answers questions about a pizza restaurant using customer reviews. The agent leverages **LangChain**, **Ollama**, and **ChromaDB** to provide contextual, data-driven responses based on relevant review data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Dataset Format](#dataset-format)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an AI agent that acts as an expert assistant for understanding customer feedback about a restaurant. The agent uses semantic search to find relevant reviews and generates intelligent responses using a local language model.

**Key Capabilities:**
- 🔍 Semantic search through restaurant reviews
- 🤖 Intelligent question answering using RAG
- 💾 Persistent vector storage for fast retrieval
- 🖥️ Fully local execution (no API keys required)
- 💬 Interactive command-line interface

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **AI Agent Architecture** | Autonomous question-answering agent with RAG capabilities |
| **Vector Search** | Semantic similarity search using embeddings |
| **Local LLM** | Runs entirely on your machine using Ollama (Llama 3.2) |
| **Persistent Storage** | ChromaDB vector database with automatic indexing |
| **Interactive CLI** | Simple, user-friendly command-line interface |
| **Privacy-First** | All data stays on your local machine |

---

## 🏗️ Architecture

The AI agent implements the RAG (Retrieval-Augmented Generation) pattern:
```
┌─────────────┐
│ User Question│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Vector Retrieval │ ◄── ChromaDB (mxbai-embed-large)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│Context Augmentation│ ◄── Prompt Template
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generation  │ ◄── Llama 3.2 (Ollama)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Answer      │
└─────────────────┘
```

**Flow Diagram:**
1. **User Input**: Question about restaurant reviews
2. **Retrieval**: Agent searches vector DB for top-5 relevant reviews
3. **Augmentation**: Combines retrieved context with question
4. **Generation**: LLM produces answer based on actual customer feedback
5. **Output**: Contextual, data-grounded response

---

## 📦 Prerequisites

Before running this AI agent, ensure you have:

### Required Software
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Ollama** ([Download](https://ollama.ai))

### Required Ollama Models
Pull the following models before running:
```bash
# Language model for answer generation
ollama pull llama3.2

# Embedding model for vector search
ollama pull mxbai-embed-large
```

To verify models are installed:
```bash
ollama list
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ahmedboussetta6/RAG-Based-AI-Agent-for-Restaurant-Review-Question-Answering.git
cd RAG-Based-AI-Agent-for-Restaurant-Review-Question-Answering
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import langchain; import pandas; print('All dependencies installed successfully!')"
```

---

## 💻 Usage

### Running the AI Agent
```bash
python main.py
```

### Example Interactions
```bash
-------------------------------
Ask your question (q to quit): How is the pizza crust?

Based on customer reviews, the pizza crust receives excellent feedback. 
Customers describe it as "perfectly crispy on the outside and chewy inside," 
with many praising its texture and quality...

-------------------------------
Ask your question (q to quit): What do customers say about delivery?

Customer feedback on delivery is mixed. Some reviews mention disappointment 
with long wait times, with one customer reporting a 1-hour delay when 
promised 30 minutes. However, the pizza quality itself is generally rated well...

-------------------------------
Ask your question (q to quit): q
```

### Sample Questions to Try
- "What are the most popular pizzas?"
- "How is the customer service?"
- "Are there any complaints about the restaurant?"
- "What do people think about the pepperoni pizza?"
- "Is the restaurant good for families?"

---

## 📁 Project Structure
```
RAG-Based-AI-Agent-for-Restaurant-Review-Question-Answering/
│
├── main.py                              # AI agent entry point and main loop
├── vector.py                            # Vector database setup and retriever
├── requirements.txt                     # Python dependencies
├── realistic_restaurant_reviews.csv    # Restaurant reviews dataset
├── README.md                            # This file
│
└── chroma_langchain_db/                # ChromaDB storage (auto-generated)
    ├── chroma.sqlite3
    └── [vector embeddings]
```

### File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Main application loop, handles user interaction and orchestrates the RAG pipeline |
| `vector.py` | Manages vector database initialization, document embedding, and retrieval setup |
| `requirements.txt` | Lists all Python package dependencies |
| `realistic_restaurant_reviews.csv` | Source data containing customer reviews with ratings and dates |
| `chroma_langchain_db/` | Persistent storage for vector embeddings (created automatically on first run) |

---

## 🔧 How It Works

### 1. **Data Indexing Phase** (`vector.py`)
```python
# Process reviews into vector embeddings
CSV Reviews → Document Objects → Embeddings → ChromaDB
```

- Reads reviews from CSV file
- Creates Document objects with metadata (rating, date)
- Generates vector embeddings using `mxbai-embed-large`
- Stores in ChromaDB for persistent, fast retrieval

### 2. **Agent Execution Phase** (`main.py`)
```python
# RAG pipeline execution
Question → Retrieve (top-5) → Augment (prompt) → Generate (LLM) → Answer
```

- **Retrieval**: Searches vector DB for semantically similar reviews
- **Augmentation**: Injects retrieved context into prompt template
- **Generation**: LLM produces answer grounded in actual data
- **Output**: Returns contextual, evidence-based response

### 3. **RAG Benefits**

| Traditional LLM | RAG-Based Agent |
|-----------------|-----------------|
| Limited to training data | Accesses up-to-date reviews |
| May hallucinate | Grounded in actual data |
| Generic responses | Context-specific answers |
| No source attribution | Traceable to source reviews |

---

## ⚙️ Configuration

### Agent Behavior Customization

**In `vector.py`** - Adjust retrieval:
```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Change number of reviews retrieved (default: 5)
)
```

**In `main.py`** - Modify agent prompt:
```python
template = """
you are an expert in answering questions about a pizza restaurant

here are some relevant reviews : {reviews}

here is the question to answer : {question}
"""
# Customize the agent's expertise and response style
```

**In `main.py`** - Change LLM model:
```python
model = OllamaLLM(model="llama3.2")  # Try: llama3.1, mistral, etc.
```

**In `vector.py`** - Change embedding model:
```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # Try other embedding models
```

### Performance Tuning

| Parameter | Location | Impact |
|-----------|----------|--------|
| `k` (retrieval count) | `vector.py` | More reviews = better context but slower |
| LLM model size | `main.py` | Larger = better quality but slower |
| Embedding model | `vector.py` | Affects search quality |

---

## 📊 Dataset Format

### Required CSV Structure

The `realistic_restaurant_reviews.csv` must contain these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Title` | String | Review headline | "Best pizza in town" |
| `Date` | String (YYYY-MM-DD) | Review date | "2024-03-15" |
| `Rating` | Integer (1-5) | Star rating | 5 |
| `Review` | String | Full review text | "The crust was perfectly..." |

### Sample Data
```csv
Title,Date,Rating,Review
Best pizza in town,2024-03-15,5,"The crust was perfectly crispy on the outside and chewy inside. Their signature pepperoni pizza had the perfect ratio of sauce to cheese, and the pepperoni curled up into little cups of deliciousness. Will definitely be back!"
Disappointed with service,2024-02-20,2,"While the pizza itself was decent, we waited over an hour for delivery despite being told it would be 30 minutes. When it finally arrived, it was barely warm. The flavors were good but the experience ruined it."
```

### Adding Your Own Data

1. Create a CSV file with the required columns
2. Replace `realistic_restaurant_reviews.csv`
3. Delete the `chroma_langchain_db/` folder
4. Run `python main.py` to rebuild the vector database

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

<details>
<summary><b>❌ "Model not found" error</b></summary>

**Cause**: Ollama models not installed

**Solution**:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
ollama list  # Verify installation
```
</details>

<details>
<summary><b>⏱️ Slow agent response times</b></summary>

**Possible solutions**:
- Use a smaller LLM model (e.g., `llama3.2:1b`)
- Reduce retrieval count in `vector.py` (e.g., `k=3`)
- Ensure Ollama is running locally
</details>

<details>
<summary><b>🔄 Database not updating with new reviews</b></summary>

**Solution**:
```bash
# Delete the vector database
rm -rf chroma_langchain_db/  # On macOS/Linux
# or
rmdir /s chroma_langchain_db  # On Windows

# Rebuild on next run
python main.py
```
</details>

<details>
<summary><b>🤔 Agent provides irrelevant answers</b></summary>

**Possible solutions**:
- Increase retrieval count (`k` value)
- Improve prompt template clarity
- Check if reviews in CSV are relevant to questions
- Try a different embedding model
</details>

<details>
<summary><b>📦 Import errors or missing dependencies</b></summary>

**Solution**:
```bash
pip install --upgrade -r requirements.txt
```
</details>

<details>
<summary><b>🚫 Ollama connection errors</b></summary>

**Solution**:
1. Ensure Ollama is running: `ollama serve`
2. Check if models are accessible: `ollama list`
3. Restart Ollama service
</details>

---

## 📚 Learn More About RAG

**RAG (Retrieval-Augmented Generation)** is an AI technique that enhances language model responses by:

1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the prompt with retrieved context
3. **Generating** accurate, grounded responses

### Why RAG?

| Problem | Solution |
|---------|----------|
| LLMs have outdated knowledge | RAG retrieves current information |
| LLMs can hallucinate | RAG grounds responses in real data |
| LLMs lack domain-specific knowledge | RAG uses custom knowledge bases |

### Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Explained](https://arxiv.org/abs/2005.11401)

---

## 🙏 Acknowledgments

- **LangChain** - Framework for building AI agents
- **Ollama** - Local LLM runtime
- **ChromaDB** - Vector database
- **Meta AI** - Llama 3.2 model

---

## 📧 Contact

For questions, suggestions, or collaborations:
- **GitHub**: [@AhmedBoussetta](https://github.com/ahmedboussetta6)
- **Email**: ahmed.boussetta@ensi-uma.tn

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ using Python, LangChain, and Ollama

</div>
