# Open WebUI and voice

Compose connects Open WebUI to the agent's OpenAI-compatible endpoint at `http://agent:8010/v1`. Sign in, choose `satellite-cv-agent`, and test model-information and prediction-history requests.

Voice input/output uses Open WebUI's browser Web API defaults. Allow microphone permission, open **Settings → Audio**, select the browser speech-to-text and text-to-speech engines, and enable automatic playback if desired. Chrome/Edge provide the most consistent browser speech support. Voice is optional per user and no microphone audio is stored by this application.

Required verification:

1. Ask for the current model and confirm a `get_model_info` call.
2. Ask for recent predictions and confirm `get_prediction_history`.
3. Ask a combined model-and-statistics question and confirm multiple sequential tools.
4. Attach an image and confirm the response contains the API-returned classification.
