from __future__ import annotations

import json
import os

import boto3
from langfuse import get_client

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_DISPATCH, TOOL_SPECS

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID") or "anthropic.claude-3-5-sonnet-20241022-v2:0"
REGION = os.environ.get("AWS_REGION") or "us-east-1"
MAX_TOOL_HOPS = 5  # hard stop against infinite tool-call loops

_client = None
_langfuse = None


def _bedrock_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def _langfuse_client():
    global _langfuse
    if _langfuse is None:
        _langfuse = get_client()
    return _langfuse


def _execute_tool(name: str, tool_input: dict) -> dict:
    langfuse = _langfuse_client()
    with langfuse.start_as_current_observation(
        as_type="tool", name=name, input=tool_input
    ) as observation:
        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            result = {"error": True, "message": f"Unknown tool: {name}"}
        else:
            try:
                result = fn(**tool_input)
            except Exception as exc:  # noqa: BLE001 - report failure, not a crash
                result = {"error": True, "message": f"Tool {name} raised: {exc}"}
        observation.update(
            output=result,
            level="ERROR" if result.get("error") else "DEFAULT",
        )
        return result


def run_agent_turn(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    messages: Converse-format history, e.g.
        [{"role": "user", "content": [{"text": "How many water predictions?"}]}]

    Returns (final_text, updated_messages) so the caller can keep the
    conversation going across turns.
    """
    langfuse = _langfuse_client()
    with langfuse.start_as_current_observation(
        as_type="agent", name="satellite-cv-agent", input=messages
    ) as agent_observation:
        result = _run_agent_turn(messages, langfuse)
        agent_observation.update(output=result[0])
        return result


def _run_agent_turn(messages: list[dict], langfuse) -> tuple[str, list[dict]]:
    client = _bedrock_client()
    history = list(messages)

    for _ in range(MAX_TOOL_HOPS):
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="bedrock-converse",
            model=MODEL_ID,
            input=history,
        ) as generation:
            resp = client.converse(
                modelId=MODEL_ID,
                system=[{"text": SYSTEM_PROMPT}],
                messages=history,
                toolConfig={"tools": TOOL_SPECS},
            )
            generation.update(
                output=resp.get("output"),
                usage_details=resp.get("usage"),
                metadata={"stop_reason": resp.get("stopReason"), "region": REGION},
            )

        output_message = resp["output"]["message"]
        history.append(output_message)
        stop_reason = resp["stopReason"]

        if stop_reason != "tool_use":
            # model produced a final text answer
            text_parts = [b["text"] for b in output_message["content"] if "text" in b]
            return "\n".join(text_parts), history

        # model wants one or more tools run -- execute each, feed results back
        tool_results = []
        for block in output_message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            result = _execute_tool(tool_use["name"], tool_use.get("input", {}))
            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_use["toolUseId"],
                    "content": [{"json": result}],
                    "status": "error" if result.get("error") else "success",
                }
            })

        history.append({"role": "user", "content": tool_results})

    return ("I couldn't complete this request within the allowed number of tool "
            "calls. Please try rephrasing or ask a narrower question."), history


if __name__ == "__main__":
    # quick manual smoke test:  uv run python -m agent.agent
    convo = [{"role": "user", "content": [{"text": "What model is currently deployed?"}]}]
    answer, _ = run_agent_turn(convo)
    print(answer)
