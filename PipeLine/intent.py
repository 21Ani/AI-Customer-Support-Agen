import json
import ollama

INTENTS = [
    "account_login",
    "payment_billing",
    "subscription",
    "console_hardware",
    "controller",
    "game_installation",
    "network_connection",
    "game_error",
    "refund",
    "general_support",
]


def classify_intent(message: str) -> str:
    prompt = f"""
Classify this Xbox customer message into exactly one intent.

Allowed intents:
{", ".join(INTENTS)}

Customer message:
{message}

Return only valid JSON:
{{"intent": "one_allowed_intent"}}
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        format="json",
    )

    result = json.loads(response["message"]["content"])
    intent = result.get("intent", "general_support")

    if intent not in INTENTS:
        return "general_support"

    return intent