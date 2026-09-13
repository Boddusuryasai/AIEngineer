import os
import json
from pathlib import Path

from openai import OpenAI 
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o-mini"

WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)


def safe_path(filename: str) -> Path:
    path = (WORKSPACE / filename).resolve()

    if not str(path).startswith(str(WORKSPACE)):
        raise ValueError("Access outside workspace is not allowed.")

    return path


def list_files():
    files = []

    for path in WORKSPACE.rglob("*"):
        if path.is_file():
            files.append(str(path.relative_to(WORKSPACE)))

    return files


def read_file(filename: str):
    path = safe_path(filename)

    if not path.exists():
        return f"File '{filename}' does not exist."

    if not path.is_file():
        return f"'{filename}' is not a file."

    return path.read_text(encoding="utf-8")


def write_file(filename: str, content: str):
    path = safe_path(filename)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return f"Successfully wrote {filename}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path of the file relative to workspace."
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Path of the file relative to workspace."
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete contents of the file."
                    }
                },
                "required": ["filename", "content"]
            }
        }
    }
]


def execute_tool(tool_name, arguments):

    if tool_name == "list_files":
        return list_files()

    if tool_name == "read_file":
        return read_file(arguments["filename"])

    if tool_name == "write_file":
        return write_file(
            arguments["filename"],
            arguments["content"]
        )

    return f"Unknown tool: {tool_name}"


SYSTEM_PROMPT = """
You are a coding agent working inside a workspace directory.

Your job is to modify files to satisfy the user's request.

Available tools:
- list_files
- read_file
- write_file

Understand the request, inspect files when necessary, use the appropriate
tools, inspect their results, and continue until the task is complete.

Read existing files before modifying them.

Do not access files outside the workspace.

Do not claim that a file was modified unless the write_file tool succeeded.

When the task is complete, provide a concise final response.
"""


def run_agent(user_request):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_request
        }
    ]

    while True:

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            print("\nAgent:")
            print(message.content)
            break

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print("\nACTION:", tool_name)
            print("INPUT:", arguments)

            try:
                observation = execute_tool(
                    tool_name,
                    arguments
                )
            except Exception as e:
                observation = f"Tool error: {e}"

            print("OBSERVATION:", observation)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(observation)
                }
            )


if __name__ == "__main__":

    print("ReAct Coding Agent")

    request = input("\nWhat should I do? ")

    run_agent(request)