# ARCHANGEL FORGE
Umbrella Dynamics agent factory. Intake form -> OpenRouter writes the prompt -> ElevenLabs agent created -> client recorded with 7-day trial date.

## Files
- app.py — the whole application (form, API calls, client records, admin)
- requirements.txt — Python dependencies
- render.yaml — Render deployment config

## Environment variables (set in Render dashboard)
- OPENROUTER_API_KEY
- ELEVENLABS_API_KEY
- INTAKE_PASSWORD

## URLs once deployed
- /            intake form
- /admin?key=YOUR_PASSWORD   client list
- /export?key=YOUR_PASSWORD  JSON backup download
- /health      quick check that keys are configured
