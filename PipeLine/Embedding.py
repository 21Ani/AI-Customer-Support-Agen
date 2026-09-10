from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = (
    BASE_DIR / "Data Preprocessing" / "xbox_training_data.csv"
)
VECTOR_DB_PATH = BASE_DIR / "vector_db"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "xbox_support"
BATCH_SIZE = 5_000


def load_training_data() -> pd.DataFrame:
    data = pd.read_csv(TRAINING_DATA_PATH)

    required_columns = ["clean_text", "historical_reply"]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data.dropna(
        subset=required_columns
    ).drop_duplicates(
        subset=required_columns
    ).reset_index(drop=True)

    data["clean_text"] = data["clean_text"].astype(str).str.strip()
    data["historical_reply"] = (
        data["historical_reply"].astype(str).str.strip()
    )

    data = data[
        data["clean_text"].ne("")
        & data["historical_reply"].ne("")
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError("No usable training rows were found.")

    return data


def create_vector_database() -> None:
    data = load_training_data()

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    messages = data["clean_text"].tolist()

    print(f"Creating embeddings for {len(messages)} messages...")
    embeddings = model.encode(
        messages,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted the previous vector collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    total_records = len(messages)

    for start in range(0, total_records, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_records)

        collection.add(
            ids=[
                f"xbox-{index}"
                for index in range(start, end)
            ],
            documents=messages[start:end],
            embeddings=embeddings[start:end],
            metadatas=[
                {
                    "historical_reply": reply,
                    "source": "xbox_training_data.csv",
                }
                for reply in data["historical_reply"].iloc[start:end]
            ],
        )

        print(
            f"Stored records {start + 1} to {end} "
            f"of {total_records}"
        )

    print(f"Stored {total_records} records successfully.")
    print(f"Database location: {VECTOR_DB_PATH}")


def search_similar_messages(
    message: str,
    number_of_results: int = 3,
) -> list[dict]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [message],
        normalize_embeddings=True,
    ).tolist()

    available_records = collection.count()
    result_count = min(number_of_results, available_records)

    if result_count == 0:
        return []

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append(
            {
                "customer_message": document,
                "historical_reply": metadata[
                    "historical_reply"
                ],
                "distance": distance,
            }
        )

    return matches


if __name__ == "__main__":
    create_vector_database()