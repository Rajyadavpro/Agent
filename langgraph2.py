# runner_langgraph_gemini.py
import os
import re
import sys
import json
import logging
from typing import Any, Dict, List, Tuple

import psycopg2
import paramiko
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("runner")

# ---------------- Env Vars ----------------
load_dotenv()

# Preferred auth: API key; fallback: ADC (gcloud)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# Defaults (override via .env or per-step fields from Gemini)
DEFAULT_DB_USER = os.getenv("DB_USER", "postgres")
DEFAULT_DB_PASSWORD = os.getenv("DB_PASSWORD", "postgrespassword")
DEFAULT_DB_NAME = os.getenv("DB_NAME", "mydatabase")
DEFAULT_DB_HOST = os.getenv("DB_HOST", "localhost")
DEFAULT_DB_PORT = int(os.getenv("DB_PORT", "5432"))

DEFAULT_SSH_USER = os.getenv("SSH_USER", "sshuser")
DEFAULT_SSH_PASSWORD = os.getenv("SSH_PASSWORD", "sshpassword")
DEFAULT_SSH_HOST = os.getenv("SSH_HOST", "localhost")
DEFAULT_SSH_PORT = int(os.getenv("SSH_PORT", "2222"))

ALLOW_DESTRUCTIVE_SQL = os.getenv("ALLOW_DESTRUCTIVE_SQL", "no").lower() in ("1", "true", "yes", "y")

# ---------------- Utilities ----------------
def read_instruction_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Instruction file not found at '{path}'. "
            f"Create it or pass a different path via: python {os.path.basename(__file__)} your_file.txt"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ---------------- Safety Checks ----------------
def is_destructive_sql(sql: str) -> bool:
    lowered = f" {sql.strip().lower()} "
    keywords = [" delete ", " drop ", " truncate ", " update ", " alter "]
    return any(kw in lowered for kw in keywords)

def is_destructive_ssh(command: str) -> bool:
    lowered = command.strip().lower()
    risky_patterns = [
        "rm ", "rm -", "rmdir", "mkfs", "shutdown", "reboot",
        "systemctl stop", "systemctl disable", "kill ", "killall"
    ]
    return any(p in lowered for p in risky_patterns)

def require_confirmation(prompt: str) -> bool:
    """Ask user to confirm destructive action."""
    resp = input(f"{prompt} [y/N]: ").strip().lower()
    return resp in ("y", "yes")

# ---------------- Normalization ----------------
def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default

def normalize_step(raw: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """
    Strict schema:
    - SQL: {"type":"sql","sql":"...","host","port","user","password","database"}
    - SSH: {"type":"ssh","command":"...","host","port","user","password"}
    Fills defaults when fields are missing or null.
    """
    if not isinstance(raw, dict):
        return False, {}, "Step is not an object"

    step_type = (raw.get("type") or raw.get("step_type") or raw.get("action") or "").strip().lower()
    if step_type not in ("sql", "ssh"):
        return False, {}, f"Unknown or missing step type: {step_type!r}"

    if step_type == "sql":
        sql = raw.get("sql") or raw.get("query") or raw.get("statement")
        if not sql or not isinstance(sql, str) or not sql.strip():
            return False, {}, "SQL step missing 'sql'/'query'/'statement'"
        norm = {
            "type": "sql",
            "sql": sql.strip(),
            "host": raw.get("host") or DEFAULT_DB_HOST,
            "port": _int_or_default(raw.get("port"), DEFAULT_DB_PORT),
            "user": raw.get("user") or DEFAULT_DB_USER,
            "password": raw.get("password") or DEFAULT_DB_PASSWORD,
            "database": raw.get("database") or DEFAULT_DB_NAME,
        }
        return True, norm, ""

    if step_type == "ssh":
        cmd = raw.get("command") or raw.get("cmd")
        if not cmd or not isinstance(cmd, str) or not cmd.strip():
            return False, {}, "SSH step missing 'command'"
        norm = {
            "type": "ssh",
            "command": cmd.strip(),
            "host": raw.get("host") or DEFAULT_SSH_HOST,
            "port": _int_or_default(raw.get("port"), DEFAULT_SSH_PORT),
            "user": raw.get("user") or DEFAULT_SSH_USER,
            "password": raw.get("password") or DEFAULT_SSH_PASSWORD,
        }
        return True, norm, ""

    return False, {}, "Unreachable branch"

def validate_and_normalize_steps(steps: Any) -> List[Dict[str, Any]]:
    if not isinstance(steps, list):
        raise ValueError("Parsed steps are not a JSON array.")
    normalized: List[Dict[str, Any]] = []
    for i, raw in enumerate(steps, 1):
        ok, norm, why = normalize_step(raw)
        if not ok:
            logger.warning(f"Skipping invalid step #{i}: {why} | raw={raw}")
            continue
        normalized.append(norm)
    if not normalized:
        raise ValueError("No valid steps found after normalization.")
    return normalized

# ---------------- LLM (Gemini) ----------------
def build_llm():
    """
    Prefer GOOGLE_API_KEY for simplicity; otherwise rely on ADC (gcloud).
    If ADC fails with RefreshError, user should run:
      gcloud auth application-default login
    """
    try:
        if GOOGLE_API_KEY:
            logger.info("Using GOOGLE_API_KEY for Gemini auth.")
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        else:
            logger.info("GOOGLE_API_KEY not set; attempting ADC (gcloud) for Gemini auth.")
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    except Exception as e:
        logger.error("Failed to initialize Gemini client: %s", e)
        raise

def extract_steps_with_gemini(instruction: str, llm) -> List[Dict[str, Any]]:
    prompt = f"""
You are a precise extraction engine.

Extract ordered steps from the instruction below. Return ONLY a valid JSON array (no backticks, no prose).
Each element must be one of:

SQL step:
{{
  "type": "sql",
  "sql": "SELECT * FROM table;",
  "host": "optional-host",
  "port": 5432,
  "user": "optional-user",
  "password": "optional-password",
  "database": "optional-db"
}}

SSH step:
{{
  "type": "ssh",
  "command": "ls -l /var/www",
  "host": "optional-host",
  "port": 2222,
  "user": "optional-user",
  "password": "optional-password"
}}

Instruction:
\"\"\"{instruction}\"\"\"
"""
    try:
        resp = llm.invoke(prompt)  # raw string => Human message for Gemini
    except Exception as e:
        msg = str(e)
        if "Reauthentication is needed" in msg or "application-default login" in msg:
            logger.error("Gemini auth failed via ADC. Run: gcloud auth application-default login")
        raise

    text = (getattr(resp, "content", None) or "").strip()
    logger.info("🔍 Gemini raw output:\n%s", text)

    # Try to extract JSON array robustly (strip fences if present)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        match = re.search(r"\[[\s\S]*\]", text)
        json_str = match.group(0) if match else text
        steps = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"JSON parse error from Gemini output: {e}\n--- RAW ---\n{text}\n-----------")

    return validate_and_normalize_steps(steps)

# ---------------- Executors ----------------
def execute_sql_command(step: Dict[str, Any]) -> Dict[str, Any]:
    sql = step["sql"]

    if is_destructive_sql(sql):
        if not require_confirmation(f"⚠️ Destructive SQL detected: {sql}\nProceed?"):
            msg = "User denied destructive SQL execution."
            logger.warning("⛔ %s", msg)
            return {"status": "skipped", "reason": msg, "sql": sql}

    try:
        logger.info("Executing SQL → %s", sql)
        conn = psycopg2.connect(
            host=step.get("host", DEFAULT_DB_HOST),
            port=step.get("port", DEFAULT_DB_PORT),
            database=step.get("database", DEFAULT_DB_NAME),
            user=step.get("user", DEFAULT_DB_USER),
            password=step.get("password", DEFAULT_DB_PASSWORD),
        )
        with conn.cursor() as cur:
            cur.execute(sql)
            if sql.strip().lower().startswith("select"):
                rows = cur.fetchall()
                logger.info("✅ SQL result rows: %d", len(rows))
                return {"status": "success", "result": rows}
            conn.commit()
        logger.info("✅ SQL executed successfully (no result set)")
        return {"status": "success", "message": "SQL executed successfully"}
    except Exception as e:
        logger.error("❌ SQL execution error: %s", e)
        return {"status": "error", "message": str(e), "sql": sql}

def execute_ssh_command(step: Dict[str, Any]) -> Dict[str, Any]:
    cmd = step["command"]

    if is_destructive_ssh(cmd):
        if not require_confirmation(f"⚠️ Destructive SSH command detected: {cmd}\nProceed?"):
            msg = "User denied destructive SSH execution."
            logger.warning("⛔ %s", msg)
            return {"status": "skipped", "reason": msg, "command": cmd}

    try:
        logger.info("Executing SSH → %s", cmd)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=step.get("host", DEFAULT_SSH_HOST),
            port=step.get("port", DEFAULT_SSH_PORT),
            username=step.get("user", DEFAULT_SSH_USER),
            password=step.get("password", DEFAULT_SSH_PASSWORD),
        )
        stdin, stdout, stderr = client.exec_command(cmd)
        output = stdout.read().decode(errors="ignore")
        error = stderr.read().decode(errors="ignore")
        client.close()
        logger.info("✅ SSH output (first 200 chars): %s", (output or error)[:200].replace("\n", " "))
        return {"status": "success", "output": output.strip(), "error": error.strip()}
    except Exception as e:
        logger.error("❌ SSH execution error: %s", e)
        return {"status": "error", "message": str(e)}

# ---------------- LangGraph Nodes ----------------
def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = build_llm()
    steps = extract_steps_with_gemini(state["instruction"], llm)
    state["steps"] = steps
    state["index"] = 0
    state["results"] = []
    return state

def step_node(state: Dict[str, Any]) -> Dict[str, Any]:
    step = state["steps"][state["index"]]
    if step["type"] == "sql":
        result = execute_sql_command(step)
    elif step["type"] == "ssh":
        result = execute_ssh_command(step)
    else:
        result = {"status": "error", "message": f"Unknown step type: {step.get('type')!r}"}

    state["results"].append({"step": step, "result": result})
    state["index"] += 1
    return state

def check_done(state: Dict[str, Any]):
    if state.get("index", 0) >= len(state.get("steps", [])):
        return END
    return "step"

def build_graph():
    workflow = StateGraph(dict)
    workflow.add_node("plan", plan_node)
    workflow.add_node("step", step_node)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "step")
    workflow.add_conditional_edges("step", check_done)
    return workflow.compile()

# ---------------- Main ----------------
def main():
    instruction_path = "instruction.txt"
    if len(sys.argv) >= 2:
        instruction_path = sys.argv[1]

    instruction_text = read_instruction_file(instruction_path)
    if not instruction_text.strip():
        raise ValueError("Instruction file is empty.")

    graph = build_graph()
    final_state = graph.invoke({"instruction": instruction_text})

    print("\n================= FINAL RESULTS =================")
    print(json.dumps(final_state.get("results", []), indent=2, ensure_ascii=False))
    print("=================================================\n")

if __name__ == "__main__":
    main()
