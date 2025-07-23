# AIDA/aida_agent/ingest.py
# This script reads documents from the runbooks directory, splits them,
# creates vector embeddings, and stores them in a persistent ChromaDB.
import os
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIDA.Ingest")

# --- Constants ---
# The path to our runbooks, as seen from inside the Docker container.
RUNBOOKS_PATH = "runbooks"
# The directory where the persistent vector database will be stored.
# This is mapped to a Docker volume in docker-compose.yml.
CHROMA_PERSIST_DIR = "/data/chroma_db"
# This is a highly-rated, lightweight, and fast sentence-transformer model.
# It runs entirely on the CPU.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# The name for our collection inside the vector store.
CHROMA_COLLECTION_NAME = "aida_runbooks"


def main():
    """
    Main function to load, split, embed, and store documents.
    """
    if not os.path.exists(RUNBOOKS_PATH) or not os.listdir(RUNBOOKS_PATH):
        logger.error(f"Runbooks directory '{RUNBOOKS_PATH}' is empty or does not exist. Aborting ingestion.")
        return

    logger.info("Starting runbook ingestion process...")

    # 1. Load Documents from the runbooks directory
    loader = DirectoryLoader(
        RUNBOOKS_PATH,
        glob="**/*.md", # Look for all markdown files in all subdirectories
        show_progress=True,
        use_multithreading=True,
        loader_kwargs={"autodetect_encoding": True}
    )
    documents = loader.load()
    if not documents:
        logger.warning("No markdown documents found. Exiting.")
        return
    logger.info(f"Loaded {len(documents)} documents.")

    # 2. Split Documents into smaller chunks for better search accuracy.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    # 3. Initialize the Embedding Model
    # This model will be downloaded from Hugging Face automatically the first time.
    # The ~/.cache/huggingface volume in docker-compose ensures it's not re-downloaded every time.
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Explicitly use CPU
    )
    logger.info("Embedding model loaded.")

    # 4. Create and Persist the Vector Store
    # This will create the ChromaDB collection on disk. If it already exists,
    # running this again will add new/updated documents.
    logger.info(f"Creating and persisting vector store at: {CHROMA_PERSIST_DIR}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )

    logger.info("Ingestion complete. Vector store has been persisted.")
    logger.info(f"Total vectors in store: {vector_store._collection.count()}")


if __name__ == "__main__":
    main()