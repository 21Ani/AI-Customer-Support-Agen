import csv
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from sklearn.metrics import accuracy_score, classification_report, f1_score

from PipeLine.main import build_support_graph

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
RESULTS = ROOT / "Evaluation" / "evaluation_results.csv"
app = Flask(__name__, static_folder=str(FRONTEND))


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND, filename)


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a support question."}), 400
    try:
        result = build_support_graph().invoke({"user_message": message, "conversation_history": []})
        return jsonify({"intent": result.get("intent", "general_support"), "reply": result.get("draft_reply", ""), "escalation_required": result.get("escalation_required", False), "escalation_reason": result.get("escalation_reason", "")})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.get("/api/evaluation")
def evaluation():
    if not RESULTS.exists():
        return jsonify({"error": "Evaluation results have not been generated yet."}), 404
    with RESULTS.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    expected = [row["Intent"] for row in rows]
    predicted = [row["Predicted_Intent"] for row in rows]
    report = classification_report(expected, predicted, output_dict=True, zero_division=0)
    report_rows = [{"intent": intent, "precision": values["precision"], "recall": values["recall"], "f1": values["f1-score"], "support": values["support"]} for intent, values in report.items() if intent not in {"accuracy", "macro avg", "weighted avg"}]
    def average(name):
        return sum(float(row.get(name, 0) or 0) for row in rows) / len(rows)
    return jsonify({"metrics": {"accuracy": accuracy_score(expected, predicted), "macro_f1": f1_score(expected, predicted, average="macro", zero_division=0), "helpfulness": average("Helpfulness"), "relevance": average("Relevance"), "groundedness": average("Groundedness"), "correctness": average("Correctness")}, "report": report_rows, "rows": rows})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)