"""
ARCHANGEL FORGE — Umbrella Dynamics agent factory
Takes an intake form, writes a custom agent prompt via OpenRouter,
provisions an ElevenLabs conversational agent, and records the client
with a 7-day trial start date.

Environment variables required (set these in Render, never in code):
  OPENROUTER_API_KEY   - your OpenRouter key (sk-or-...)
  ELEVENLABS_API_KEY   - your ElevenLabs key
  INTAKE_PASSWORD      - password required to submit the intake form
Optional:
  ADMIN_PASSWORD       - password for /admin (defaults to INTAKE_PASSWORD)
  OPENROUTER_MODEL     - model id (default: anthropic/claude-sonnet-4.5)
  CLIENTS_FILE         - where client records are stored (default: clients.json)
  TRIAL_DAYS           - trial length in days (default: 7)
"""

import os
import json
import datetime
import requests
from flask import Flask, request, render_template_string, Response

app = Flask(__name__)

# ---------------- configuration ----------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
INTAKE_PASSWORD = os.environ.get("INTAKE_PASSWORD", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", INTAKE_PASSWORD)
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
CLIENTS_FILE = os.environ.get("CLIENTS_FILE", "clients.json")
TRIAL_DAYS = int(os.environ.get("TRIAL_DAYS", "7"))

BRAND = {
    "green": "#21c063",
    "teal": "#0b5c4d",
    "ink": "#202b27",
    "mist": "#eef1ee",
    "line": "#d9e2dc",
}

# ---------------- client records ----------------

def load_clients():
    try:
        with open(CLIENTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_client(record):
    clients = load_clients()
    clients.append(record)
    with open(CLIENTS_FILE, "w") as f:
        json.dump(clients, f, indent=2)


# ---------------- OpenRouter: write the agent's brain ----------------

def generate_agent_prompt(fields):
    instructions = f"""You are writing the system prompt for an AI voice agent that answers
phone calls for a small business. The agent's jobs: answer warmly and professionally,
find out why the caller is calling, detect and politely dismiss spam/robocalls/cold sales
pitches, and forward genuine customers or take a detailed message.

Business details:
- Business name: {fields['business_name']}
- What the business does: {fields['description']}
- Business hours: {fields['hours']}
- Forward genuine calls to: {fields['forward_name']} 
- Preferred greeting (use it if provided): {fields['greeting'] or 'none provided, write a natural one'}
- Questions the agent should ask callers: {fields['questions'] or 'use sensible defaults for this business type'}
- Things the agent must NEVER say or do: {fields['never_say'] or 'nothing specific'}
- How to handle spam: {fields['spam_policy'] or 'politely end the call and ask to be removed from their list'}
- Language: {fields['language']}

Write ONLY the system prompt text itself, ready to be given to the voice agent.
It should be thorough but concise, written in second person ("You are...").
Include: the greeting to use, how to identify the caller's purpose, spam detection
behavior, what information to collect from genuine callers, and how to close calls.
"""
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": instructions}],
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------- ElevenLabs: build the voice agent ----------------

def create_elevenlabs_agent(name, system_prompt, first_message, language):
    resp = requests.post(
        "https://api.elevenlabs.io/v1/convai/agents/create",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "name": name,
            "conversation_config": {
                "agent": {
                    "prompt": {"prompt": system_prompt},
                    "first_message": first_message,
                    "language": language,
                }
            },
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json().get("agent_id", "unknown")


# ---------------- shared page style ----------------
STYLE = """
<style>
  :root{--green:#21c063;--teal:#0b5c4d;--ink:#202b27;--mist:#eef1ee;--line:#d9e2dc;--soft:#57665f}
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;background:var(--mist);color:var(--ink);
       min-height:100vh;padding:28px 16px 64px}
  .card{max-width:560px;margin:0 auto;background:#fff;border:1px solid var(--line);border-radius:20px;
        padding:30px 26px;box-shadow:0 18px 50px rgba(6,63,53,.10)}
  h1{font-size:1.45rem;margin-bottom:4px;color:var(--teal)}
  .sub{color:var(--soft);font-size:.92rem;margin-bottom:22px}
  label{display:block;font-weight:600;font-size:.88rem;margin:16px 0 6px}
  label small{font-weight:400;color:var(--soft)}
  input,textarea,select{width:100%;padding:12px 14px;border:1.5px solid var(--line);border-radius:12px;
        font-family:inherit;font-size:.95rem;background:#fafcfa}
  input:focus,textarea:focus,select:focus{outline:none;border-color:var(--green)}
  textarea{min-height:76px;resize:vertical}
  button{width:100%;margin-top:24px;padding:15px;border:none;border-radius:999px;background:var(--green);
        color:#fff;font-size:1rem;font-weight:700;cursor:pointer}
  button:active{background:#17a552}
  .ok{background:rgba(33,192,99,.12);border:1px solid var(--green);border-radius:12px;padding:14px;margin:14px 0}
  .err{background:#fdeeee;border:1px solid #e0a49a;border-radius:12px;padding:14px;margin:14px 0;color:#8f3a2b;
       word-break:break-word;font-size:.88rem}
  .kv{font-size:.92rem;margin:6px 0}.kv b{color:var(--teal)}
  table{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:14px}
  th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}
  a{color:var(--teal)}
</style>
"""

# ---------------- intake form ----------------
FORM_HTML = STYLE + """
<div class="card">
  <h1>ARCHANGEL FORGE</h1>
  <p class="sub">Umbrella Dynamics · New agent intake. Fill this once — a working AI receptionist comes out the other side.</p>
  {% if error %}<div class="err"><b>Something went wrong:</b><br>{{ error }}</div>{% endif %}
  <form method="POST" action="/create">
    <label>Intake password</label>
    <input type="password" name="password" required>

    <label>Business name</label>
    <input name="business_name" required placeholder="Keller Plumbing">

    <label>What the business does <small>(one or two sentences)</small></label>
    <textarea name="description" required placeholder="Residential plumbing repair and installation in the Denver area."></textarea>

    <label>Business hours</label>
    <input name="hours" required placeholder="Mon–Fri 8am–6pm, Sat 9am–1pm">

    <label>Who genuine calls go to <small>(name the agent should mention)</small></label>
    <input name="forward_name" required placeholder="Dana">

    <label>Custom greeting <small>(optional — leave blank and the Forge writes one)</small></label>
    <input name="greeting" placeholder="Good morning, you've reached Keller Plumbing...">

    <label>Questions the agent should ask callers <small>(optional)</small></label>
    <textarea name="questions" placeholder="What's the issue? What's your address? Is this urgent?"></textarea>

    <label>Things the agent should never say or do <small>(optional)</small></label>
    <textarea name="never_say" placeholder="Never quote exact prices. Never promise same-day service."></textarea>

    <label>How to handle spam calls <small>(optional)</small></label>
    <input name="spam_policy" placeholder="Politely end the call and block">

    <label>Agent language</label>
    <select name="language">
      <option value="en">English</option>
      <option value="es">Spanish</option>
    </select>

    <label>Referral code <small>(optional — how did this client find us?)</small></label>
    <input name="ref_code" placeholder="mike">

    <button type="submit">⚡ Forge this agent</button>
  </form>
</div>
"""

SUCCESS_HTML = STYLE + """
<div class="card">
  <h1>Agent forged ✓</h1>
  <p class="sub">The AI receptionist for <b>{{ business }}</b> is live.</p>
  <div class="ok">
    <div class="kv"><b>Agent ID:</b> {{ agent_id }}</div>
    <div class="kv"><b>Trial started:</b> {{ trial_start }}</div>
    <div class="kv"><b>Trial ends:</b> {{ trial_end }} ({{ trial_days }} days)</div>
  </div>
  <p class="sub">Next: open the ElevenLabs dashboard to attach a phone number / WhatsApp
  to this agent and place a test call. The client record has been saved.</p>
  <p><a href="/">← Forge another agent</a></p>
</div>
"""

ADMIN_HTML = STYLE + """
<div class="card">
  <h1>Clients</h1>
  <p class="sub">{{ count }} record(s). <a href="/export?key={{ key }}">Download JSON backup</a></p>
  <table>
    <tr><th>Business</th><th>Trial ends</th><th>Ref</th><th>Agent ID</th></tr>
    {% for c in clients %}
    <tr><td>{{ c.business_name }}</td><td>{{ c.trial_end }}</td>
        <td>{{ c.ref_code or "—" }}</td><td style="font-size:.7rem">{{ c.agent_id }}</td></tr>
    {% endfor %}
  </table>
</div>
"""


# ---------------- routes ----------------

@app.route("/")
def index():
    return render_template_string(FORM_HTML, error=None)


@app.route("/create", methods=["POST"])
def create():
    if request.form.get("password", "") != INTAKE_PASSWORD or not INTAKE_PASSWORD:
        return render_template_string(FORM_HTML, error="Wrong intake password."), 403

    fields = {
        "business_name": request.form.get("business_name", "").strip(),
        "description": request.form.get("description", "").strip(),
        "hours": request.form.get("hours", "").strip(),
        "forward_name": request.form.get("forward_name", "").strip(),
        "greeting": request.form.get("greeting", "").strip(),
        "questions": request.form.get("questions", "").strip(),
        "never_say": request.form.get("never_say", "").strip(),
        "spam_policy": request.form.get("spam_policy", "").strip(),
        "language": request.form.get("language", "en"),
        "ref_code": request.form.get("ref_code", "").strip(),
    }

    try:
        system_prompt = generate_agent_prompt(fields)
        first_message = fields["greeting"] or (
            f"Hello, you've reached {fields['business_name']}. How can I help you today?"
        )
        agent_id = create_elevenlabs_agent(
            name=f"UD - {fields['business_name']}",
            system_prompt=system_prompt,
            first_message=first_message,
            language=fields["language"],
        )
    except requests.HTTPError as e:
        detail = e.response.text[:600] if e.response is not None else str(e)
        return render_template_string(
            FORM_HTML, error=f"API error ({e.response.status_code if e.response is not None else '?'}): {detail}"
        ), 502
    except Exception as e:
        return render_template_string(FORM_HTML, error=str(e)[:600]), 500

    today = datetime.date.today()
    trial_end = today + datetime.timedelta(days=TRIAL_DAYS)
    record = {
        **fields,
        "agent_id": agent_id,
        "trial_start": today.isoformat(),
        "trial_end": trial_end.isoformat(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        save_client(record)
    except Exception:
        pass  # never lose a created agent over a bookkeeping failure

    return render_template_string(
        SUCCESS_HTML,
        business=fields["business_name"],
        agent_id=agent_id,
        trial_start=record["trial_start"],
        trial_end=record["trial_end"],
        trial_days=TRIAL_DAYS,
    )


@app.route("/admin")
def admin():
    key = request.args.get("key", "")
    if not ADMIN_PASSWORD or key != ADMIN_PASSWORD:
        return "Not authorized. Use /admin?key=YOUR_ADMIN_PASSWORD", 403
    clients = load_clients()
    return render_template_string(ADMIN_HTML, clients=clients, count=len(clients), key=key)


@app.route("/export")
def export():
    key = request.args.get("key", "")
    if not ADMIN_PASSWORD or key != ADMIN_PASSWORD:
        return "Not authorized.", 403
    return Response(
        json.dumps(load_clients(), indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=clients-backup.json"},
    )


@app.route("/health")
def health():
    return {"status": "ok", "keys_set": bool(OPENROUTER_API_KEY and ELEVENLABS_API_KEY and INTAKE_PASSWORD)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
