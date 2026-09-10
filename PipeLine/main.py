from typing import TypedDict

import ollama
from langgraph.graph import END, StateGraph

from PipeLine.Embedding import search_similar_messages
from PipeLine.intent import classify_intent


OLLAMA_MODEL = "qwen2.5:3b"
MAX_DISTANCE = 0.9

ESCALATION_KEYWORDS = [
    "hacked",
    "hack",
    "stolen",
    "fraud",
    "security",
    "refund",
    "lawsuit",
    "legal",
    "threat",
    "harassment",
    "charged twice",
    "payment dispute",
]


class SupportState(TypedDict, total=False):
    user_message: str
    intent: str
    retrieved_examples: list[dict]
    escalation_required: bool
    escalation_reason: str
    draft_reply: str
    conversation_history: list[dict]


def receive_message(state: SupportState) -> SupportState:
    message = state["user_message"]
    history = state.get("conversation_history", []).copy()

    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return {
        "conversation_history": history,
    }


def classify_message(state: SupportState) -> SupportState:
    intent = classify_intent(state["user_message"])

    return {
        "intent": intent,
    }


def retrieve_examples(state: SupportState) -> SupportState:
    try:
        examples = search_similar_messages(
            state["user_message"],
            number_of_results=3,
        )
    except Exception as error:
        print(f"Retrieval warning: {error}")
        examples = []

    return {
        "retrieved_examples": examples,
    }


def escalation_router(state: SupportState) -> SupportState:
    message = state["user_message"].lower()
    examples = state.get("retrieved_examples", [])

    keyword_match = next(
        (
            keyword
            for keyword in ESCALATION_KEYWORDS
            if keyword in message
        ),
        None,
    )

    if keyword_match:
        return {
            "escalation_required": True,
            "escalation_reason": (
                f"The message contains a sensitive issue: "
                f"'{keyword_match}'."
            ),
        }

    if not examples:
        return {
            "escalation_required": True,
            "escalation_reason": (
                "No similar historical Xbox support examples "
                "were found."
            ),
        }

    best_distance = examples[0].get("distance")

    if best_distance is not None and best_distance > MAX_DISTANCE:
        return {
            "escalation_required": True,
            "escalation_reason": (
                "The closest historical examples are not similar "
                "enough to trust for automatic handling."
            ),
        }

    return {
        "escalation_required": False,
        "escalation_reason": "",
    }


def route_after_escalation_check(state: SupportState) -> str:
    if state.get("escalation_required", False):
        return "human_handoff"

    return "generate_reply"


def human_handoff(state: SupportState) -> SupportState:
    print("\nHuman escalation required.")
    print(f"Reason: {state['escalation_reason']}")
    print("A human support agent should review this conversation.\n")

    return {
        "draft_reply": (
            "This issue needs to be reviewed by a human support agent."
        ),
    }


def generate_reply(state: SupportState) -> SupportState:
    examples_text = "\n\n".join(
        [
            (
                f"Customer example: {example['customer_message']}\n"
                f"Historical response: {example['historical_reply']}"
            )
            for example in state.get("retrieved_examples", [])
        ]
    )

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in state.get("conversation_history", [])
    )

    prompt = f"""
You are an Xbox customer-support assistant.

Current customer message:
{state['user_message']}

Intent:
{state['intent']}

Conversation:
{history_text}

Historical examples for guidance:
{examples_text}

Write a new response for the current customer.

Important rules:
- Use historical responses only as guidance.
- Do not copy any historical response word-for-word.
- Do not mention or refer to the historical examples.
- Do not use agent initials or signatures such as ^JA.
- Do not give a generic answer unrelated to the current message.
- Directly address the customer's specific problem.
- Do not invent policies, refunds, links, or guarantees.
- Keep the response concise and helpful.
- Return only the final customer-facing response.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate an original customer-support response. "
                    "Never copy retrieved examples verbatim."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.4,
        },
    )

    reply = response["message"]["content"].strip()

    # Remove common agent signatures if the model still includes one.
    reply = reply.replace("^JA", "").strip()

    return {
        "draft_reply": reply,
    }


def finish_turn(state: SupportState) -> SupportState:
    history = state.get("conversation_history", []).copy()

    history.append(
        {
            "role": "assistant",
            "content": state["draft_reply"],
        }
    )

    return {
        "conversation_history": history,
    }


def build_support_graph():
    graph = StateGraph(SupportState)

    graph.add_node("receive_message", receive_message)
    graph.add_node("classify_message", classify_message)
    graph.add_node("retrieve_examples", retrieve_examples)
    graph.add_node("escalation_router", escalation_router)
    graph.add_node("human_handoff", human_handoff)
    graph.add_node("generate_reply", generate_reply)
    graph.add_node("finish_turn", finish_turn)

    graph.set_entry_point("receive_message")

    graph.add_edge("receive_message", "classify_message")
    graph.add_edge("classify_message", "retrieve_examples")
    graph.add_edge("retrieve_examples", "escalation_router")

    graph.add_conditional_edges(
        "escalation_router",
        route_after_escalation_check,
        {
            "human_handoff": "human_handoff",
            "generate_reply": "generate_reply",
        },
    )

    graph.add_edge("human_handoff", "finish_turn")
    graph.add_edge("generate_reply", "finish_turn")
    graph.add_edge("finish_turn", END)

    return graph.compile()


def run_support_agent() -> None:
    app = build_support_graph()
    conversation_history: list[dict] = []

    print("Xbox Support Agent")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not user_message:
            print("Please enter a message.\n")
            continue

        result = app.invoke(
            {
                "user_message": user_message,
                "conversation_history": conversation_history,
            }
        )

        conversation_history = result.get(
            "conversation_history",
            conversation_history,
        )

        print(f"\nIntent: {result.get('intent', 'unknown')}")
        print(f"Xbox Agent: {result['draft_reply']}\n")


if __name__ == "__main__":
    run_support_agent()