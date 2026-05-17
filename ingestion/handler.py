"""
Lambda handler — scrapes AWS documentation pages and indexes them into
OpenSearch Serverless using Bedrock Titan Embeddings.

Invoke payload (all optional):
{
  "urls": ["https://docs.aws.amazon.com/..."],   # override default seed list
  "max_pages": 200                                # crawl depth limit
}
"""
import os
import json
import boto3
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

REGION = os.environ["OSS_REGION"]
OSS_ENDPOINT = os.environ["OSS_ENDPOINT"].replace("https://", "")
OSS_INDEX = os.environ["OSS_INDEX"]
EMBED_MODEL = os.environ["BEDROCK_EMBED_MODEL"]
DOCS_BUCKET = os.environ["DOCS_BUCKET"]

# AWS documentation seed pages — covers a broad range of services
DEFAULT_SEED_URLS = [
    "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
    "https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html",
    "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
    "https://docs.aws.amazon.com/dynamodb/latest/developerguide/Introduction.html",
    "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html",
    "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
    "https://docs.aws.amazon.com/opensearch-service/latest/developerguide/what-is.html",
    "https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html",
    "https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/sqs/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/sns/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-what-is.html",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (AWS Docs Bot Indexer)"}
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _get_vector_store(embeddings) -> OpenSearchVectorSearch:
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        refreshable_credentials=credentials,
        region=REGION,
        service="aoss",
    )
    return OpenSearchVectorSearch(
        opensearch_url=f"https://{OSS_ENDPOINT}",
        index_name=OSS_INDEX,
        embedding_function=embeddings,
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        engine="faiss",
        timeout=120,
    )


def _scrape_page(url: str) -> tuple[str, list[str]]:
    """Returns (text_content, list_of_same_domain_links)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception:
        return "", []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav/footer noise
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("div", {"id": "main-content"}) or soup
    text = main.get_text(separator=" ", strip=True)

    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        parsed = urlparse(href)
        if parsed.netloc == urlparse(url).netloc and parsed.path.endswith(".html"):
            links.append(href.split("#")[0])  # strip anchors

    return text, links


def lambda_handler(event, context):
    seed_urls = event.get("urls", DEFAULT_SEED_URLS)
    max_pages = event.get("max_pages", 150)

    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL, region_name=REGION)
    store = _get_vector_store(embeddings)

    visited = set()
    queue = deque(seed_urls)
    documents: list[Document] = []
    s3 = boto3.client("s3")

    print(f"Starting ingestion — seed={len(seed_urls)} pages, max={max_pages}")

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        print(f"Scraping [{len(visited)}/{max_pages}]: {url}")
        text, links = _scrape_page(url)

        if len(text) < 200:
            continue

        # Store raw text in S3
        key = "raw/" + url.replace("https://", "").replace("/", "_") + ".txt"
        s3.put_object(Bucket=DOCS_BUCKET, Key=key, Body=text.encode())

        chunks = SPLITTER.split_text(text)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"source": url},
            ))

        # Enqueue discovered same-domain links
        for link in links:
            if link not in visited:
                queue.append(link)

        # Batch-index every 10 docs to avoid bulk payload timeouts
        if len(documents) >= 10:
            store.add_documents(documents)
            print(f"Indexed batch of {len(documents)} chunks")
            documents = []

    if documents:
        store.add_documents(documents)
        print(f"Indexed final batch of {len(documents)} chunks")

    summary = {"pages_scraped": len(visited), "status": "complete"}
    print(json.dumps(summary))
    return summary
