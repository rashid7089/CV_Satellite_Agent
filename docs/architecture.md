# Architecture

```mermaid
flowchart LR
  U[Browser] --> F[Nginx frontend]
  F --> B[FastAPI backend]
  W[Open WebUI chat and voice] --> A[Bedrock agent adapter]
  A --> L[AWS Bedrock LLM]
  A --> B
  B --> M[Keras or ONNX model]
  B --> P[(PostgreSQL)]
  B --> R[(Redis rate limits)]
  B -. optional .-> S[(S3 object storage)]
```

The model loads once during backend startup. A successful inference is persisted with its hash, Top-K result, latency, model version, and optional S3 URI. The agent obtains operational answers only through backend tools. PostgreSQL, Redis, and Open WebUI use persistent Docker volumes.
