import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.agent import build_agent

app = FastAPI(title="AWS Docs Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent_executor = None


def get_agent():
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor


def get_dynamo():
    return boto3.resource("dynamodb", region_name=os.environ["OSS_REGION"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Load conversation history from DynamoDB
    table = get_dynamo().Table(os.environ["DYNAMO_TABLE"])
    history_resp = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(session_id),
        ScanIndexForward=True,
        Limit=10,
    )
    history = history_resp.get("Items", [])
    history_text = "\n".join(
        f"Human: {h['human']}\nAssistant: {h['assistant']}" for h in history
    )

    full_input = f"{history_text}\nHuman: {req.message}" if history_text else req.message

    try:
        agent = get_agent()
        result = agent.invoke({"input": full_input})
        answer = result.get("output", "I couldn't find an answer.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Persist turn to DynamoDB
    table.put_item(Item={
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "human": req.message,
        "assistant": answer,
        "ttl": int(datetime.now(timezone.utc).timestamp()) + 86400 * 7,  # 7-day TTL
    })

    return ChatResponse(answer=answer, session_id=session_id)


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    table = get_dynamo().Table(os.environ["DYNAMO_TABLE"])
    resp = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(session_id),
    )
    with table.batch_writer() as batch:
        for item in resp.get("Items", []):
            batch.delete_item(Key={"session_id": item["session_id"], "timestamp": item["timestamp"]})
    return {"deleted": True}
