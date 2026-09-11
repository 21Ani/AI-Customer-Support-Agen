# Xbox Customer Support Agent

An AI customer-support agent built from real Xbox Twitter-support conversations. It classifies customer messages, retrieves similar historical conversations, drafts grounded replies, and escalates sensitive or low-confidence cases.

## Requirements

- Python 3.10+
- Ollama
- Git

## Install Ollama

Download Ollama from [ollama.com/download](https://ollama.com/download).

Then download the local model:

```powershell
ollama pull qwen2.5:3b
ollama list
```

## Setup

```powershell
git clone https://github.com/21Ani/AI-Customer-Support-Agen.git
cd AI-Customer-Support-Agen
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Dataset

Download the Customer Support on Twitter dataset from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place the source file at:

```text
Data Preprocessing/twcs.csv
```

The processed training file must be available at:

```text
Data Preprocessing/xbox_training_data.csv
```

## Build the Vector Database

```powershell
python PipeLine\Embedding.py
```

This creates the local `vector_db/` Chroma database. Rebuild it whenever the training data changes.

## Run the Agent

Command-line interface:

```powershell
python -m PipeLine.main
```

Web interface:

```powershell
python -m flask --app PipeLine.server run --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser.

The web interface includes an Xbox support chat and a **Golden eval** button that displays the evaluation metrics and the contents of `Evaluation/evaluation_results.csv`.

## Run Evaluation

The golden set is stored at `Evaluation/Golden_Test_Set.csv`. Run:

```powershell
python -u -m Evaluation.Eval
```

Results are written to `Evaluation/evaluation_results.csv` and include predicted intents, generated replies, and reply-quality scores.

## System Workflow

```text
Customer question
	-> intent classification
	-> retrieve similar Xbox conversations
	-> escalation check
	-> grounded reply or human handoff
```

## Project Structure

```text
Data Preprocessing/   preprocessing notebook and local datasets
Evaluation/            golden set, evaluation harness, and results
PipeLine/              intent, embeddings, agent, and web server
frontend/              chat interface and Xbox background image
requirements.txt       Python dependencies
```

## Decision Log

1. Xbox was selected as the target brand because it has enough support conversations.
2. A small set of intents was chosen instead of modelling every dataset label.
3. Ollama was selected for free local LLM inference.
4. `qwen2.5:3b` is used for intent classification and reply generation.
5. `all-MiniLM-L6-v2` is used for local sentence embeddings.
6. ChromaDB is used for local vector search.
7. Customer messages are embedded; historical replies are stored as metadata.
8. Historical replies guide generation but should not be copied verbatim.
9. Sensitive topics are escalated to reduce risk.
10. Low retrieval similarity causes escalation instead of unsupported generation.
11. Macro-F1 is reported because intent classes may be imbalanced.
12. A manually labelled golden set is used as the evaluation reference.
13. Reply quality is scored with an LLM-as-judge rubric.
14. Human review is required to measure agreement with the automated judge.
15. Local datasets and generated vector databases are excluded from Git because of size.

## Limitations

- The source dataset is not included because it is very large.
- Conversation history is stored in memory during a session.
- The local vector database must be rebuilt when training data changes.
- The initial golden set is smaller than the required final evaluation size.
- Using the same local model for generation and judging can introduce bias.