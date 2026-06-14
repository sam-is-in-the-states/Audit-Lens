# RevenueLens

RevenueLens is an AI-powered revenue recognition review system for SaaS contracts. The project evaluates whether a multi-agent architecture can improve the accuracy, consistency, and explainability of revenue recognition reviews compared to a traditional single-prompt LLM approach.

The system analyzes SaaS contracts, retrieves relevant accounting guidance and internal policies, evaluates revenue recognition treatment, identifies accounting risks, generates revenue schedules, and produces an auditable review memo.

## Features

* PDF contract ingestion and parsing
* Retrieval-Augmented Generation (RAG) using ChromaDB
* Single-prompt baseline implementation
* Structured JSON output generation
* Revenue schedule calculations
* Accounting risk identification
* Multi-agent architecture (planned)
* Evaluation framework for comparing baseline and multi-agent performance

## Project Structure

```text
revenue-lens/
├── app/
├── tests/
├── requirements.txt
├── .env
└── README.md
```

## Prerequisites

* Python 3.12
* OpenAI API Key

## Installation

### Create a Virtual Environment

```bash
py -3.12 -m venv venv (Windows)
```

### Activate the Virtual Environment

#### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)

```cmd
venv\Scripts\activate.bat
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root using the .env.example:


## Running the Project

```bash
python app/main.py
```

## Roadmap

### Phase 1

* Single-prompt baseline
* PDF parsing
* RAG pipeline
* Revenue schedule generation

### Phase 2

* Contract Understanding Agent
* Accounting Standards Agent
* Audit/Risk Agent
* Financial Impact Agent
* Decision/Synthesis Agent

### Phase 3

* Evaluation framework
* Baseline vs multi-agent comparison
* Performance and accuracy analysis
