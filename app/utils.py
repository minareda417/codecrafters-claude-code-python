import subprocess
import json
import shlex
from dataclasses import dataclass
import re
from pathlib import Path
import os
from urllib.parse import urlparse
from urllib.request import url2pathname

from .lsp_client import LSPClient

from openai import OpenAI
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

LSP = {
    "type": "function",
    "function": {
        "name": "LSP",
        "description": "Returns the file, line, and character where a symbol is defined.",
        "parameters": {
        "type": "object",
        "properties": {
            "operation": {
            "type": "string",
            "enum": ["goToDefinition"]
            },
            "filePath": {
            "type": "string",
            "description": "Absolute path, or a path relative to the agent's current working directory."
            },
            "line": {
            "type": "integer",
            "minimum": 1,
            "description": "1-based line number of the symbol."
            },
            "character": {
            "type": "integer",
            "minimum": 1,
            "description": "1-based character offset in UTF-16 code units (same as character count for ASCII)."
            }
        },
        "required": ["operation", "filePath", "line", "character"]
        }
    }
    }

URI = set()

TOOLS = [READ, WRITE, BASH, SKILL, LSP]

def getFileContent(filePath: str)->str:
    with open(filePath, 'r') as f:
        return f.read()

def execute_tool(tool_call: ChatCompletionMessageFunctionToolCall, client: OpenAI, LSPClient: LSPClient):
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
        return _skill(arguments, client)
    elif function_name == "LSP":
        return _LSP(arguments, LSPClient)

def _read(arguments: dict) -> str:
    return getFileContent(arguments["file_path"])

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

def _skill(arguments: dict, client: OpenAI) -> str:
    name = arguments["name"]
    raw_args = arguments.get("args", "")
    args = shlex.split(raw_args) if raw_args else []
    skill = build_skill(name, args)
    if skill.fork:
        subagent_messages = [{"role": "user", "content": skill.content}]
        sub_chat = client.chat.completions.create(model = "anthropic/claude-haiku-4.5",
                                                  messages = subagent_messages,
                                                  tools = TOOLS)
        return f"Skill {skill.name} ran in a separate context and returned: {sub_chat.choices[0].message.content}"
    return skill.content

def _LSP(arguments: dict, lspClient: LSPClient) -> str:
    operation = arguments["operation"]
    if operation == "goToDefinition":
        return goToDefinition(arguments, lspClient)
    elif operation == "findReferences ":
        return findReferences(arguments, lspClient)
    else:
        raise NotImplementedError

def getAbsolutePAth(filePath: str) -> str:
    return str(Path(filePath).resolve()) if not Path(filePath).is_absolute() else filePath

def didOpen(filePath: str, uri: str, lspClient: LSPClient):
    text = getFileContent(getAbsolutePAth(filePath))
    lspClient.notify(method="textDocument/didOpen", params={"textDocument":{"uri": uri, "languageId": "python", "version": 1, "text": text}})
    URI.add(uri)

def uriToFileName(uri: str) -> str:
    return str(Path(url2pathname(urlparse(uri).path))).split('/')[-1]

def getUri(filePath: str, lspClient: LSPClient) -> str:
    uri = Path(getAbsolutePAth(filePath)).as_uri()
    if uri not in URI:
        didOpen(filePath, uri, lspClient)
        URI.add(uri)
    return uri

def goToDefinition(arguments: dict, lspCLient: LSPClient) -> str:
    uri = getUri(arguments["filePath"], lspCLient)
    line = arguments["line"]-1
    character = arguments["character"]-1
    definition = lspCLient.request(method="textDocument/definition", params={"textDocument": {"uri": uri}, "position": {"line": line, "Character": character}})
    if not definition:
        return "Definition not found"
    response_uri = uriToFileName(definition["uri"]) 
    start = definition["range"]["start"]
    return f"Defined in {str(response_uri)} at {start["line"]+1}:{start["character"]+1}"

def getReferencesDict(references) -> dict:
    references_dict = {}
    for reference in references:
        uri = reference["uri"]
        ref_path = uriToFileName(reference["uri"])
        start_line = reference["range"]["start"]["line"]+1
        start_character = reference["range"]["start"]["character"]+1
        if references_dict[ref_path]:
            references_dict[ref_path].append((start_line, start_character))
        else:
            references_dict[ref_path] = [(start_line, start_character)]
    return references_dict

def findReferences(arguments: dict, lspCLient: LSPClient) -> str:
    print("Hello")
    uri = getUri(arguments["filePath"], lspCLient)
    line = arguments["line"]-1
    character = arguments["character"]-1
    references = lspCLient.request("textDocument/references", {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}, "context": {"includeDeclaration": True}})
    if not references:
        return "Refrences not found"
    references_dict = getReferencesDict(references)
    print(references_dict)
    ans = f"Found {len(references)} across {len(references_dict.keys)} files:\n\n"
    for fileName, locations in references:
        ans += f"{fileName}:\n"
        for location in locations:
            ans += f"\tLine {locations[0]}:{location[1]}\n"
        ans += "\n"
    return ans 




@dataclass
class skill:
    name: str
    description: str
    content: str
    fork: bool

def _replace_placeholders(line: str, arguments: list[str], skill: str) -> str:
    line = re.sub(r"`([^`]+/[^`]*)`", lambda m: f"`.claude/skills/{skill}/{m.group(1)}`",line)
    if not arguments:
        return line
    line = line.replace("$ARGUMENTS", " ".join(arguments))
    line = re.sub(r"\$ARGUMENT\[(\d+)\]", lambda m: str(arguments[int(m.group(1))]), line)
    line = re.sub(r"\$(\d+)", lambda m: str(arguments[int(m.group(1))]), line)
    return line

def build_skill(skill_name: str, arguments: list[str]) -> skill:
    file_path = f".claude/skills/{skill_name}/SKILL.md"
    description = ""
    content = ""
    with open(file_path, 'r') as f:
        for _ in range(2):
            f.readline()
        description += f.readline()[13:]
        l = f.readline()
        fork = False
        if l == "context: fork\n":
            fork = True
            for _ in range(2):
                f.readline()
        else:
            f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            content += _replace_placeholders(line, arguments, skill_name)
        f.close()
    return skill(skill_name, description, content, fork)

def process_skills(skills: list[str], arguments: str) -> list[skill]:
    SKILLS = []
    for skill in skills:
        SKILLS.append(build_skill(skill, arguments))
    return SKILLS

def get_skills_messages(p: str) -> list[dict]:
    command = p.split()
    skills = []
    idx = 0
    while idx < len(command) and command[idx][0] == '/' :
        skills.append(command[idx][1:])
        idx += 1
    if idx <= len(command) - 1:
        arguments = command[idx:]
    else:
        arguments = []
    SKILLS = process_skills(skills, arguments)
    messages = [{"role": "system", "content": f"You have access to the following skills:\n\n- {"\n".join(f"{skill.name}: {skill.description}" for skill in SKILLS)}\n"}]
    for skill in SKILLS:
        messages.append({"role": "user", "content": skill.content})
    return messages

def get_messages(p: str) -> list[dict]:
    if p[0] == '/':
        return get_skills_messages(p)
    else:
        messages = []
        skills_path = Path(".claude/skills")
        if skills_path.exists():
            skills = [skill.name for skill in skills_path.iterdir() if skill.is_dir()]
            SKILLS = process_skills(skills, p)
            messages.append({"role": "system", "content": f"You have access to the following skills:\n\n{"\n".join(f"{skill.name}: {skill.description}" for skill in SKILLS)}\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."})
        messages.append({"role": "user", "content": p})
        return messages

def agent_loop(client: OpenAI, p: str):
    messages = get_messages(p)
    lspClient = LSPClient.from_environment()

    while True:
        # print(messages)
        response = client.chat.completions.create(model = "anthropic/claude-haiku-4.5",
                                                  messages = messages,
                                                  tools = TOOLS)
        
        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("no choices in response")
        
        message = response.choices[0].message
        messages.append(message)
        tool_calls = message.tool_calls

        if tool_calls is None:
            print(message.content)
            break

        for tool_call in tool_calls:
            result = execute_tool(tool_call, client, lspClient)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})