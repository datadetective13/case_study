import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.tools import tool


def _get_vector_store() -> OpenSearchVectorSearch:
    region = os.environ["OSS_REGION"]
    endpoint = os.environ["OSS_ENDPOINT"].replace("https://", "")
    index = os.environ["OSS_INDEX"]
    embed_model = os.environ["BEDROCK_EMBED_MODEL"]

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        refreshable_credentials=credentials,
        region=region,
        service="aoss",
    )

    embeddings = BedrockEmbeddings(
        model_id=embed_model,
        region_name=region,
    )

    return OpenSearchVectorSearch(
        opensearch_url=f"https://{endpoint}",
        index_name=index,
        embedding_function=embeddings,
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        engine="faiss",
    )


@tool
def search_aws_docs(query: str) -> str:
    """Search AWS documentation to answer questions about AWS services, features, pricing, and best practices."""
    try:
        store = _get_vector_store()
        docs = store.similarity_search(query, k=5)
        if not docs:
            return "No relevant documentation found for this query."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "AWS Documentation")
            results.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error searching documentation: {str(e)}"


@tool
def get_aws_service_overview(service_name: str) -> str:
    """Get a high-level overview of an AWS service including its key features and use cases."""
    query = f"{service_name} overview features use cases getting started"
    return search_aws_docs.invoke(query)


@tool
def compare_aws_services(services: str) -> str:
    """Compare two or more AWS services. Input should be comma-separated service names (e.g. 'Lambda, ECS, EC2')."""
    query = f"comparison between {services} differences when to use"
    return search_aws_docs.invoke(query)


TOOLS = [search_aws_docs, get_aws_service_overview, compare_aws_services]
