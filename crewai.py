import os
import re
import json
import logging
import psycopg2
import paramiko
from dotenv import load_dotenv
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Env Vars ----------------
load_dotenv()
DEFAULT_DB_USER = os.getenv("DB_USER", "postgres")
DEFAULT_DB_PASSWORD = os.getenv("DB_PASSWORD", "postgrespassword")
DEFAULT_DB_NAME = os.getenv("DB_NAME", "mydatabase")
DEFAULT_SSH_USER = os.getenv("SSH_USER", "sshuser")
DEFAULT_SSH_PASSWORD = os.getenv("SSH_PASSWORD", "sshpassword")

# ---------------- SQL ----------------
def execute_sql_command(state: Dict[str, Any]):
    sql = state["step"]["sql"]
    db_info = state["step"]
    try:
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
        logger.info("✅ SQL executed successfully")
        return {"status": "success", "message": "Executed"}
    except Exception as e:
        logger.error(f"❌ SQL error: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- SSH ----------------
def execute_ssh_command(state: Dict[str, Any]):
    ssh_info = state["step"]
    try:
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
            logger.warning(f"⚠️ SSH error: {error.strip()}")
        return {"status": "success", "output": output.strip(), "error": error.strip()}
    except Exception as e:
        logger.error(f"❌ SSH error: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- JSON Extract ----------------
def extract_steps(instruction: str, llm):
    prompt = f"""
    Extract ordered steps from the instruction below.
    Return ONLY JSON array of objects of type 'sql' or 'ssh'.

    Instruction:
    \"\"\"{instruction}\"\"\"
    """
    parsed = llm.invoke(prompt)  # ✅ Gemini accepts raw string input
    text = parsed.content.strip()
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"JSON parse error: {e}\n{text}")

# ---------------- Build Graph ----------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

workflow = StateGraph(dict)

def plan_node(state: Dict[str, Any]):
    steps = extract_steps(state["instruction"], llm)
    state["steps"] = steps
    state["index"] = 0
    state["results"] = []
    return state

def step_node(state: Dict[str, Any]):
    step = state["steps"][state["index"]]
    state["step"] = step
    if step["type"] == "sql":
        result = execute_sql_command(state)
    elif step["type"] == "ssh":
        result = execute_ssh_command(state)
    else:
        result = {"status": "error", "message": "Unknown step type"}
    state["results"].append(result)
    state["index"] += 1
    return state

def check_done(state: Dict[str, Any]):
    if state["index"] >= len(state["steps"]):
        return END
    return "step"

workflow.add_node("plan", plan_node)
workflow.add_node("step", step_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "step")
workflow.add_conditional_edges("step", check_done)

graph = workflow.compile()

# ---------------- Run from file ----------------
if __name__ == "__main__":
    with open("instructions.txt", "r") as f:
        instruction_text = f.read().strip()

    final_state = graph.invoke({"instruction": instruction_text})
    print("✅ Final Results:", final_state["results"])
