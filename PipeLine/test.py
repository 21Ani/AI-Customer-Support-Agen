from intent import classify_intent


test_messages = [
    "My Xbox controller will not connect",
    "I cannot log in to my Xbox account",
    "I was charged twice for my subscription",
    "My game will not download",
]


for message in test_messages:
    print(f"Classifying: {message}", flush=True)

    try:
        intent = classify_intent(message)
        print(f"Intent: {intent}", flush=True)
    except Exception as error:
        print(f"Error: {error}", flush=True)

    print("-" * 40, flush=True)