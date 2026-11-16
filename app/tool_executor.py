import subprocess
import json
from .skill import build_skill
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall

READ = {
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
    }

WRITE =     {
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

BASH = {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                "type": "string",
                "description": "The command to execute"
                }
            }
            }
        }
    }

SKILL = {
        "type": "function",
        "function": {
        "name": "Skill",
        "description": "Load a skill's instructions into the conversation",
        "parameters": {
            "type": "object",
            "required": ["name"],
            "properties": {
            "name": { "type": "string", "description": "The name of the skill to use" },
            "args": { "type": "string", "description": "Optional arguments for the skill" }
            }
        }
        }
    }

TOOLS = [READ, WRITE, BASH, SKILL]


def execute_tool(tool_call: ChatCompletionMessageFunctionToolCall):
    function = tool_call.function
    function_name = function.name
    arguments = json.loads(function.arguments)
    if function_name == "Read":
        return _read(arguments)
    elif function_name == "Write":
        return _write(arguments)
    elif function_name == "Bash":
        return _bash(arguments)
    elif function_name == "Skill":
        return _skill(arguments)

def _read(arguments: dict) -> str:
    with open(arguments["file_path"], 'r') as f:
        return f.read()

def _write(arguments: dict) -> str:
    content = arguments["content"]
    with open(arguments["file_path"], 'w') as f:
        f.write(content)
        f.close()
    return content

def _bash(arguments: dict) -> str:
    result = subprocess.run(args=arguments["command"], shell= True, capture_output=True, text=True)
    if result.returncode != 0:
        return result.stderr
    return result.stdout

def _skill(arguments: dict) -> str:
    print(arguments)
    name = arguments["name"]
    args = arguments.get("args", "")
    skill = build_skill(name, args)
    return skill.content