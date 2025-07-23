# AIDA/aida_agent/tools.py
import logging
from langchain.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Setup ---
logger = logging.getLogger("AIDA.Tools")

# --- Constants for RAG Tool ---
CHROMA_PERSIST_DIR = "/data/chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "aida_runbooks"

# --- Tool Definitions ---

@tool
def search_runbooks(query: str) -> str:
    """
    Searches the AIDA technical runbooks for procedures and diagnostic information.
    Use this tool FIRST to find standard troubleshooting steps for specific alerts
    like 'HighCpuUsage' or 'CrashLoopBackOff'. The query should be a concise
    description of the problem you are trying to solve.
    """
    try:
        # Initialize the same embedding model used for ingestion
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        # Load the persisted vector store
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )

        # Create a retriever to find the most relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Get top 3 results

        logger.info(f"Searching runbooks with query: '{query}'")
        results = retriever.get_relevant_documents(query)

        if not results:
            return "No relevant documents found in the runbooks for this query."

        # Format the results into a single string for the agent
        return "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        logger.error(f"An error occurred during runbook search: {e}", exc_info=True)
        return f"Error: Could not perform search. Details: {e}"


@tool
def query_prometheus(promql_query: str) -> str:
    """
    (MOCKED) Executes a PromQL query against the Prometheus monitoring system to get metrics.
    Use this to get data about CPU, memory, latency, or error rates.
    Example query: 'rate(http_requests_total{job="api"}[5m])'
    """
    logger.info(f"Executing MOCKED Prometheus query: '{promql_query}'")

    # --- Mocked Responses ---
    if "HighCpuUsage" in promql_query:
        return """
        Mocked Prometheus Response:
        [
            {"metric": {"pod": "billing-service-5c687d7f9-x7v9w"}, "value": [1678886400, "0.95"]},
            {"metric": {"pod": "frontend-6c7b8d8f9-a1b2c"}, "value": [1678886400, "0.15"]}
        ]
        This data indicates that pod 'billing-service-5c687d7f9-x7v9w' has very high CPU usage (95%).
        """
    return "Mocked Prometheus Response: No specific data for this query. Try a query related to the 'HighCpuUsage' alert."


@tool
def kubectl_tool(command: str) -> str:
    """
    (MOCKED) Executes a safe, read-only kubectl command on the Kubernetes cluster.
    Use this to inspect the state of pods, services, and deployments.
    Allowed commands start with 'get', 'describe'.
    Example commands: 'describe pod billing-service-5c687d7f9-x7v9w', 'get events --sort-by=.metadata.creationTimestamp'
    """
    logger.info(f"Executing MOCKED kubectl command: 'kubectl {command}'")

    # --- Mocked Responses ---
    if "describe pod billing-service-5c687d7f9-x7v9w" in command:
        return """
        Mocked Kubectl Response:
        Name:         billing-service-5c687d7f9-x7v9w
        Namespace:    production
        Status:       Running
        ...
        Containers:
          billing-service:
            Image:      company/billing-service:v2.1.4
            Ports:      8080/TCP
            ...
            Environment:
              DATABASE_URL:           <set to the value of a secret>
              DOWNSTREAM_API_HOST:    http://payment-processor-svc:8000
            ...
        Events:
          Type     Reason     Age   From     Message
          ----     ------     ----  ----     -------
          Warning  Unhealthy  5m    kubelet  Readiness probe failed: Get "http://:8080/healthz": context deadline exceeded
          Normal   Starting   25m   kubelet  Starting container billing-service
        """
    return "Mocked Kubectl Response: Command not recognized or no specific output available for this mock."

# A list that we can easily import into our main agent file.
all_tools = [search_runbooks, query_prometheus, kubectl_tool]