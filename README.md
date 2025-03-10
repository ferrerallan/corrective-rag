# Corrective RAG

An advanced Retrieval Augmented Generation system with self-correction features built using LangGraph.

## Overview

This project implements a robust question-answering system that combines document retrieval, web search, and LLM generation with built-in evaluation loops to ensure accurate, grounded responses.


## Features

- **Question Routing**: Automatically chooses between local vectorstore and web search
- **Document Evaluation**: Filters for relevance before generating answers
- **Hallucination Detection**: Verifies generated content against source documents
- **Answer Validation**: Ensures responses actually address the initial question
- **Self-Correction**: Reruns generation or switches to web search when needed

## Installation

```bash
git clone https://github.com/ferrerallan/corrective-rag
cd corrective-rag
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with:

```
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## Usage

```python
from graph import app

# Basic usage
response = app.invoke({"question": "How does LangGraph work?"})
print(response["generation"])

# Access individual components
from graph.nodes import retrieve, grade_documents, generate, web_search
```

## Project Structure

- `graph.py`: Main workflow definition and conditional logic
- `state.py`: State data structure shared between nodes
- `nodes/`: Core processing components (retrieve, grade, generate, search)
- `chains/`: LLM chains for evaluation and routing

## Implementation Notes

- Configured to route questions about agents, prompt engineering, and adversarial attacks to the local vectorstore
- Uses OpenAI models for generation and evaluation tasks
- Web search powered by Tavily Search API
- Uses LangGraph for workflow orchestration
