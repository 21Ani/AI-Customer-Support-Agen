import json
import time
from pathlib import Path

import ollama
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from PipeLine.main import build_support_graph


AGENT_MODEL = "qwen2.5:3b"
JUDGE_MODEL = "qwen2.5:3b"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Golden_Test_Set.csv"
OUTPUT_PATH = BASE_DIR / "evaluation_results.csv"


def parse_scores(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    scores = json.loads(content)

    score_names = [
        "helpfulness",
        "relevance",
        "groundedness",
        "correctness",
    ]

    result = {}

    for score_name in score_names:
        if score_name not in scores:
            raise ValueError(
                f"Missing score: {score_name}"
            )

        result[score_name] = max(
            1,
            min(5, int(scores[score_name])),
        )

    return result


def judge_reply(
    user_question: str,
    expected_reply: str,
    generated_reply: str,
) -> dict:
    prompt = f"""
Evaluate the generated Xbox customer-support reply.

Customer question:
{user_question}

Expected human-written reply:
{expected_reply}

Generated agent reply:
{generated_reply}

Give a score from 1 to 5 for each category:

- helpfulness: Does it help solve the customer's problem?
- relevance: Does it directly address the question?
- groundedness: Is it supported by the expected support behavior?
- correctness: Is the information accurate and safe?

Return only valid JSON. Do not include markdown:

{{
    "helpfulness": 1,
    "relevance": 1,
    "groundedness": 1,
    "correctness": 1
}}
"""

    for attempt in range(3):
        try:
            response = ollama.chat(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluator of customer-support "
                            "replies. Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                format="json",
                options={
                    "temperature": 0,
                },
            )

            content = response["message"]["content"]

            if not content:
                raise ValueError(
                    "Ollama returned an empty response."
                )

            return parse_scores(content)

        except Exception as error:
            if attempt == 2:
                raise RuntimeError(
                    f"Ollama judge failed after 3 attempts: {error}"
                ) from error

            wait_seconds = 3 * (attempt + 1)

            print(
                f"Judge failed: {error}",
                flush=True,
            )
            print(
                f"Retrying in {wait_seconds} seconds...",
                flush=True,
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Ollama judge failed.")


def run_evaluation() -> None:
    print("Evaluation started...", flush=True)
    print(f"Reading file: {DATA_PATH}", flush=True)

    data = pd.read_csv(
        DATA_PATH,
        encoding="cp1252",
    )

    required_columns = [
        "User_Question",
        "Intent",
        "Reply",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    print(
        f"Loaded {len(data)} evaluation examples.",
        flush=True,
    )

    app = build_support_graph()

    predicted_intents = []
    generated_replies = []
    judge_scores = []

    for index, row in data.iterrows():
        question = str(row["User_Question"])
        expected_reply = str(row["Reply"])

        print(
            f"\nEvaluating {index + 1}/{len(data)}",
            flush=True,
        )
        print(
            f"Question: {question}",
            flush=True,
        )

        try:
            result = app.invoke(
                {
                    "user_message": question,
                    "conversation_history": [],
                }
            )

            predicted_intent = result.get(
                "intent",
                "general_support",
            )

            generated_reply = result.get(
                "draft_reply",
                "",
            )

        except Exception as error:
            print(
                f"Agent error: {error}",
                flush=True,
            )

            predicted_intent = "general_support"
            generated_reply = (
                "The agent could not generate a reply."
            )

        predicted_intents.append(predicted_intent)
        generated_replies.append(generated_reply)

        print(
            f"Predicted intent: {predicted_intent}",
            flush=True,
        )
        print(
            "Judging generated reply...",
            flush=True,
        )

        try:
            scores = judge_reply(
                user_question=question,
                expected_reply=expected_reply,
                generated_reply=generated_reply,
            )

        except Exception as error:
            print(
                f"Judge error: {error}",
                flush=True,
            )

            scores = {
                "helpfulness": 0,
                "relevance": 0,
                "groundedness": 0,
                "correctness": 0,
            }

        judge_scores.append(scores)

    data["Predicted_Intent"] = predicted_intents
    data["Generated_Reply"] = generated_replies

    data["Helpfulness"] = [
        scores["helpfulness"]
        for scores in judge_scores
    ]

    data["Relevance"] = [
        scores["relevance"]
        for scores in judge_scores
    ]

    data["Groundedness"] = [
        scores["groundedness"]
        for scores in judge_scores
    ]

    data["Correctness"] = [
        scores["correctness"]
        for scores in judge_scores
    ]

    accuracy = accuracy_score(
        data["Intent"],
        data["Predicted_Intent"],
    )

    macro_f1 = f1_score(
        data["Intent"],
        data["Predicted_Intent"],
        average="macro",
        zero_division=0,
    )

    print("\n===== INTENT RESULTS =====")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            data["Intent"],
            data["Predicted_Intent"],
            zero_division=0,
        )
    )

    print("\n===== REPLY RESULTS =====")

    print(
        f"Average helpfulness: "
        f"{data['Helpfulness'].mean():.2f}/5"
    )

    print(
        f"Average relevance: "
        f"{data['Relevance'].mean():.2f}/5"
    )

    print(
        f"Average groundedness: "
        f"{data['Groundedness'].mean():.2f}/5"
    )

    print(
        f"Average correctness: "
        f"{data['Correctness'].mean():.2f}/5"
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nDetailed results saved to: {OUTPUT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    run_evaluation()