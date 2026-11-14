import argparse
import os
import sys
import json
import subprocess

from openai import OpenAI

from .tool_executor import tools, execute_tool

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "Read",
#             "description": "Read and return the contents of a file",
#             "parameters": {
#             "type": "object",
#             "properties": {
#                 "file_path": {
#                 "type": "string",
#                 "description": "The path to the file to read"
#                 }
#             },
#             "required": ["file_path"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "Write",
#             "description": "Write content to a file",
#             "parameters": {
#             "type": "object",
#             "required": ["file_path", "content"],
#             "properties": {
#                 "file_path": {
#                 "type": "string",
#                 "description": "The path of the file to write to"
#                 },
#                 "content": {
#                 "type": "string",
#                 "description": "The content to write to the file"
#                 }
#             }
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "Bash",
#             "description": "Execute a shell command",
#             "parameters": {
#             "type": "object",
#             "required": ["command"],
#             "properties": {
#                 "command": {
#                 "type": "string",
#                 "description": "The command to execute"
#                 }
#             }
#             }
#         }
#     }
# ]

# def _read(arguments: dict) -> str:
#     with open(arguments["file_path"], 'r') as f:
#             return f.read()

# def _write(arguments: dict) -> str:
#     content = arguments["content"]
#     with open(arguments["file_path"], 'w') as f:
#         f.write(content)
#         f.close()
#     return content

# def _bash(arguments: dict) -> str:
#     command = arguments["command"].split()
#     result = subprocess.run(args=command, capture_output=True, text=True)
#     if result.returncode == 1:        
#         return result.stderr
#     return result.stdout

# def _execute_tool(tool_call):
#     function = tool_call.function
#     function_name = function.name
#     arguments = json.loads(function.arguments)
#     if function_name == "Read":
#         return _read(arguments)
#     elif function_name == "Write":
#         return _write(arguments)
#     elif function_name == "Bash":
#         return _bash(arguments)

def agent_loop(client: OpenAI, p):
    messages=[{"role": "user", "content": p}]

    while True:
        response = client.chat.completions.create(model = "anthropic/claude-haiku-4.5",
                                                  messages = messages,
                                                  tools = tools)
        
        if not response.choices or len(response.choices) == 0:
                raise RuntimeError("no choices in response")
        
        # print("Logs from your program will appear here!", file=sys.stderr)

        message = response.choices[0].message
        messages.append(message)
        tool_calls = message.tool_calls

        if tool_calls is None:
            print(message.content)
            break

        for tool_call in tool_calls:
            result = execute_tool(tool_call)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    agent_loop(client, args.p)


if __name__ == "__main__":
    main()
