import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

tools = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                "type": "string",
                "description": "The path to the file to read"
                }
            },
            "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
            "type": "object",
            "required": ["file_path", "content"],
            "properties": {
                "file_path": {
                "type": "string",
                "description": "The path of the file to write to"
                },
                "content": {
                "type": "string",
                "description": "The content to write to the file"
                }
            }
            }
        }
}
]

def _read(file_path: str) -> str:
    with open(file_path, 'r') as f:
            return f.read()

def _write(file_path: str, content: str):
    with open(file_path, 'w') as f:
        f.write(content)

def _execute_tool(tool_call):
    function = tool_call.function
    function_name = function.name
    if function_name == "Read":
        function_arguments = json.loads(function.arguments)
        return _read(function_arguments["file_path"])
    elif function_name == "Write":
        print(function)
        arguments = json.loads(function.arguments)
        file_path = arguments["file_path"]
        content = arguments["content"]
        _write(file_path, content)
        return content


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
            result = _execute_tool(tool_call)
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
