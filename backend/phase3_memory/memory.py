# Long term memory: a local Chroma vector store built from transactions.json,
# so the agent can retrieve relevant past spending rather than only reasoning
# about the current purchase. Chroma's default embedding function runs a
# small model locally, no API key or paid service involved.
import json
from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_data"


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name="transactions")


def _transaction_to_text(transaction):
    return (
        f"{transaction['date']}: {transaction['merchant']}, {transaction['category']}, "
        f"Rs {transaction['amount']}, paid with card {transaction['card_used']}"
    )


def ensure_populated(collection):
    if collection.count() > 0:
        return

    with open(DATA_DIR / "transactions.json") as data_file:
        transactions = json.load(data_file)

    collection.add(
        ids=[str(index) for index in range(len(transactions))],
        documents=[_transaction_to_text(transaction) for transaction in transactions],
        metadatas=transactions,
    )
