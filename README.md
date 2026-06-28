# RevenueLens

RevenueLens is an AI-powered revenue recognition review system for SaaS contracts. The project evaluates whether a multi-agent architecture can improve the accuracy, consistency, and explainability of revenue recognition reviews compared to a traditional single-prompt Large Language Model (LLM) approach.

The system analyzes SaaS contracts, retrieves relevant accounting guidance and internal policies using Retrieval-Augmented Generation (RAG), evaluates revenue recognition treatment under ASC 606, identifies accounting risks, generates revenue schedules, and produces an auditable review memo.

---

# Features

* PDF contract ingestion and parsing
* Retrieval-Augmented Generation (RAG) using ChromaDB
* Local LLM inference using Ollama and Qwen3 1.7B
* Single-prompt baseline implementation
* Structured JSON output generation
* Revenue schedule generation
* Accounting risk identification
* Multi-agent architecture (planned)
* Evaluation framework comparing baseline and multi-agent systems

---

# Technology Stack

* Python 3.12.10
* Ollama
* Qwen3 1.7B
* LangChain
* LangGraph
* ChromaDB
* OpenAI Python SDK (used to communicate with the local Ollama server)
* Pydantic

---

# Prerequisites

Before running the project, install:

* Python 3.12.10
* Ollama (https://ollama.com/download)

Verify the installations:

```bash
python --version
```

```bash
ollama --version
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>

cd revenue-lens
```

---

## 2. Create a Virtual Environment

Windows

```bash
py -3.12 -m venv venv
```

macOS/Linux

```bash
python3 -m venv venv
```

---

## 3. Activate the Virtual Environment

### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
venv\Scripts\activate.bat
```

### macOS/Linux

```bash
source venv/bin/activate
```

---

## 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 5. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Download the Local Model

```bash
ollama pull qwen3:1.7b-q8_0
```

Verify that the model has been installed:

```bash
ollama list
```

---

## 7. Start the Ollama Server

```bash
ollama serve
```

On Windows and macOS, Ollama typically runs automatically as a background service.

---

## 8. Configure Environment Variables

Copy the example environment file.

Windows

```bash
copy .env.example .env
```

macOS/Linux

```bash
cp .env.example .env
```

The `.env` file should contain:

```text
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3:1.7b
OPENAI_API_KEY=dummy
```

---

# Test installation

```bash
python app/llm/client.py
```

---

# Running the Project

```bash
python app/main.py
```

---

# Architecture

## Phase 1

* PDF contract parsing
* RAG pipeline
* Single-prompt baseline
* Revenue schedule generation

## Phase 2

* Contract Understanding Agent
* Accounting Standards Agent
* Audit & Risk Agent
* Financial Impact Agent
* Decision & Synthesis Agent

## Phase 3

* Evaluation framework
* Baseline vs. multi-agent comparison
* Accuracy and performance analysis
* Explainability metrics

---

# Development Workflow

Whenever a new team member sets up the project:

1. Install Python 3.12.10
2. Install Ollama
3. Clone the repository
4. Create and activate a virtual environment
5. Install project dependencies
6. Pull the Qwen3 1.7B model
7. Copy `.env.example` to `.env`
8. Start Ollama
9. Run the application

---

# Future Improvements

* Additional open-weight models
* Agent memory
* Human-in-the-loop review
* Financial statement impact visualization
* Web interface
* Automated benchmark dataset
* Cloud deployment support
