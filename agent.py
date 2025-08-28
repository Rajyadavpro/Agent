import os
import json
import re
import paramiko
import psycopg2
import logging
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.tools import ToolContext

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- Env Vars ----------------
load_dotenv()

DEFAULT_DB_USER = os.getenv("DB_USER", "postgres")
DEFAULT_DB_PASSWORD = os.getenv("DB_PASSWORD", "postgrespassword")
DEFAULT_DB_NAME = os.getenv("DB_NAME", "mydatabase")
DEFAULT_SSH_USER = os.getenv("SSH_USER", "sshuser")
DEFAULT_SSH_PASSWORD = os.getenv("SSH_PASSWORD", "sshpassword")


# ---------------- SQL ----------------
def execute_sql_command(sql, db_info):
    try:
        logger.info(f"Executing SQL → {sql}")
        conn = psycopg2.connect(
            host=db_info.get("host", os.getenv("DB_HOST", "localhost")),
            port=db_info.get("port", int(os.getenv("DB_PORT", 5432))),
            database=db_info.get("database", DEFAULT_DB_NAME),
            user=db_info.get("user", DEFAULT_DB_USER),
            password=db_info.get("password", DEFAULT_DB_PASSWORD),
        )
        with conn.cursor() as cur:
            cur.execute(sql)
            if sql.strip().lower().startswith("select"):
                result = cur.fetchall()
                logger.info(f"✅ SQL result: {result}")
                return {"status": "success", "result": result}
            conn.commit()
            logger.info("✅ SQL executed successfully (no result set)")
            return {"status": "success", "message": "SQL executed successfully"}
    except Exception as e:
        logger.error(f"❌ SQL execution error: {e}")
        return {"status": "error", "message": str(e)}


# ---------------- SSH ----------------
def execute_ssh_command(ssh_info):
    try:
        logger.info(f"Executing SSH → {ssh_info['command']}")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ssh_info.get("host", os.getenv("SSH_HOST", "localhost")),
            port=ssh_info.get("port", int(os.getenv("SSH_PORT", 2222))),
            username=ssh_info.get("user", DEFAULT_SSH_USER),
            password=ssh_info.get("password", DEFAULT_SSH_PASSWORD),
        )
        stdin, stdout, stderr = client.exec_command(ssh_info["command"])
        output = stdout.read().decode()
        error = stderr.read().decode()
        client.close()
        logger.info(f"✅ SSH output: {output.strip()}")
        if error:
            logger.warning(f"⚠️ SSH error output: {error.strip()}")
        return {"status": "success", "output": output.strip(), "error": error.strip()}
    except Exception as e:
        logger.error(f"❌ SSH execution error: {e}")
        return {"status": "error", "message": str(e)}


# ---------------- JSON Helper ----------------
def extract_json_from_text(text: str):
    """Extract JSON array from LLM output, even if wrapped in text/code fences."""
    try:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip("` \n")
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw output:\n{text}")


# ---------------- Executor ----------------
def execute_all_steps(instruction_text: str, tool_context: ToolContext):
    parsed = tool_context.llm(
        f"""Extract ordered steps from the instruction below. Return ONLY JSON like:
[
  {{
    "type": "sql",
    "sql": "...",
    "database": "...",
    "user": "...",
    "host": "...",
    "port": 5432
  }},
  {{
    "type": "ssh",
    "host": "...",
    "user": "...",
    "port": 2222,
    "command": "..."
  }}
]

Instruction:
\"\"\"{instruction_text}\"\"\""""
    )

    try:
        steps = extract_json_from_text(parsed.text)
    except Exception as e:
        logger.error(f"❌ Failed to parse steps: {e}")
        return {"error": str(e)}

    results = []
    for idx, step in enumerate(steps, start=1):
        logger.info(f"➡️ Step {idx}: {step}")
        if step["type"] == "sql":
            sql = step["sql"]
            if any(cmd in sql.lower() for cmd in ["delete", "drop", "update", "truncate"]):
                confirm = tool_context.prompt(f"⚠️ Destructive SQL: `{sql}`. Proceed? (yes/no): ")
                if confirm.strip().lower() != "yes":
                    results.append({"status": "skipped", "reason": "user denied destructive SQL"})
                    logger.warning("⏭️ Skipped destructive SQL by user choice")
                    continue
            results.append(execute_sql_command(sql, step))

        elif step["type"] == "ssh":
            results.append(execute_ssh_command(step))

    return results


# ---------------- Agent ----------------
root_agent = Agent(
    name="multi_tool_agent",
    model="gemini-2.0-flash",
    description="Executes SQL and SSH commands in order, extracted from a text instruction.",
    instruction="Your job is to extract and sequentially execute all SQL and SSH commands in the instruction file using the proper tools.",
    tools=[execute_all_steps],
)
