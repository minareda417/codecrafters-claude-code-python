import subprocess
import json
import shlex
from dataclasses import dataclass
import re
from pathlib import Path
import os
from urllib.parse import urlparse
from urllib.request import url2pathname
from collections import defaultdict

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
            "enum": ["goToDefinition", "findReferences"]
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
VERSIONS = defaultdict(int)
REPORTS = defaultdict(set)
CURSOR = 0

TOOLS = [READ, WRITE, BASH, SKILL, LSP]

def getFileContent(filePath: str)->str:
    with open(filePath, 'r') as f:
        return f.read()

def execute_tool(tool_call: ChatCompletionMessageFunctionToolCall, client: OpenAI, LSPClient: LSPClient):
    function = tool_call.function
    function_name = function.name
    arguments = json.loads(function.arguments)
    if function_name == "Read":
        return read(arguments)
    elif function_name == "Write":
        return write(arguments, LSPClient)
    elif function_name == "Bash":
        return bash(arguments)
    elif function_name == "Skill":
        return _skill(arguments, client)
    elif function_name == "LSP":
        return LSP(arguments, LSPClient)

def read(arguments: dict) -> str:
    path = Path(arguments["file_path"])
    if (path.is_file()):
        return getFileContent(arguments["file_path"])
    return f"Directory {arguments['file_path']} content:\n\n" + ' '.join([str(itemName) for itemName in path.iterdir()])

def write(arguments: dict, lspClient: LSPClient) -> str:
    content = arguments["content"]
    filePath = arguments["file_path"]
    with open(filePath, 'w') as f:
        f.write(content)
        f.close()
    if lspClient is not None:
        uri = Path(getAbsolutePAth(filePath)).as_uri()
        if uri in URI:
            lspClient.notify(method="textDocument/didChange", params={"textDocument":{"uri": uri, "languageId": "python", "version": VERSIONS[uri]+1}, "contentChanges": [{"text": content}]})
        else:
            lspClient.notify(method="textDocument/didOpen", params={"textDocument":{"uri": uri, "languageId": "python", "version": 1, "text": content}})
            URI.add(uri)
        VERSIONS[uri] += 1
    return content

def bash(arguments: dict) -> str:
    result = subprocess.run(args=arguments["command"], shell= True, capture_output=True, text=True)
    if result.returncode != 0:
        return result.stderr
    return result.stdout

def _skill(arguments: dict, client: OpenAI) -> str:
    name = arguments["name"]
    raw_args = arguments.get("args", "")
    args = shlex.split(raw_args) if raw_args else []
    skill = buildSkill(name, args)
    if skill.fork:
        subagent_messages = [{"role": "user", "content": skill.content}]
        sub_chat = client.chat.completions.create(model = "anthropic/claude-haiku-4.5",
                                                  messages = subagent_messages,
                                                  tools = TOOLS)
        return f"Skill {skill.name} ran in a separate context and returned: {sub_chat.choices[0].message.content}"
    return skill.content

def LSP(arguments: dict, lspClient: LSPClient) -> str:
    operation = arguments["operation"]
    if operation == "goToDefinition":
        return goToDefinition(arguments, lspClient)
    if operation == "findReferences":
        return findReferences(arguments, lspClient)
    raise NotImplementedError

def getAbsolutePAth(filePath: str) -> str:
    return str(Path(filePath).resolve()) if not Path(filePath).is_absolute() else filePath

def didOpen(filePath: str, uri: str, lspClient: LSPClient):
    text = getFileContent(getAbsolutePAth(filePath))
    lspClient.notify(method="textDocument/didOpen", params={"textDocument":{"uri": uri, "languageId": "python", "version": 1, "text": text}})
    VERSIONS[uri] = 1

def uriToFileName(uri: str) -> str:
    return str(Path(url2pathname(urlparse(uri).path))).split('/')[-1]

def getUri(filePath: str, lspClient: LSPClient) -> str:
    uri = Path(getAbsolutePAth(filePath)).as_uri()
    if lspClient is None:
        print("lspClient is None")
    elif uri not in URI:
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
    return f"Defined in {str(response_uri)} at {start["line"]+1}:{start["character"]+1}\n\n"

def getReferencesDict(references) -> dict:
    references_dict = defaultdict(list)
    for reference in references:
        uri = reference["uri"]
        ref_path = uriToFileName(reference["uri"])
        start_line = reference["range"]["start"]["line"]+1
        start_character = reference["range"]["start"]["character"]+1
        references_dict[ref_path].append((start_line, start_character))
    return references_dict

def formatReferences(references_dict: defaultdict, n_ref: int) -> str:
    ans = f"Found {n_ref} references across {len(references_dict.keys())} files:\n\n"
    for uri, locations in references_dict.items():
        ans += f"{uriToFileName(uri)}:\n"
        for i in range(len(locations)):
            ans += f"\tLine {locations[i][0]}:{locations[i][1]}\n"
        ans += "\n"
    return ans 

def findReferences(arguments: dict, lspCLient: LSPClient) -> str:
    uri = getUri(arguments["filePath"], lspCLient)
    line = arguments["line"]-1
    character = arguments["character"]-1
    references = lspCLient.request("textDocument/references", {"textDocument": {"uri": uri}, "position": {"line": line, "character": character}, "context": {"includeDeclaration": True}})
    if not references:
        return "Refrences not found"
    references_dict = getReferencesDict(references)
    return formatReferences(references_dict, len(references))

@dataclass(frozen=True)
class Diagnostic:
    startLine: int
    startCharacter: int
    endLine: int
    endCharacter: int
    severity: int
    message: str

def getDiagnostics(lspClient: LSPClient) -> str:
    global CURSOR
    res = lspClient.notifications(CURSOR)
    CURSOR = res["cursor"]
    notifications = res["notifications"]
    ans = ""
    for notification in notifications:
        if notification["method"] != "textDocument/publishDiagnostics":
            continue
        params = notification["params"]
        uri = params["uri"]
        version = params["version"]
        if version is not None and (uri not in VERSIONS or version != VERSIONS[uri]):
            continue
        diagnostics = params["diagnostics"]
        if not diagnostics:
            REPORTS[uri] = set()
        else:
            current = {
                Diagnostic(
                    diagnostic["range"]["start"]["line"],
                    diagnostic["range"]["start"]["character"],
                    diagnostic["range"]["end"]["line"],
                    diagnostic["range"]["end"]["character"],
                    diagnostic["severity"],
                    diagnostic["message"],
                )
                for diagnostic in params["diagnostics"]
            }
            newDiagnostics = current - REPORTS[uri]
            REPORTS[uri] = current
            if newDiagnostics:
                ans += f"<new-diagnostics>\nThe following new diagnostic issues were detected\n\n"
                for diagnostic in REPORTS[uri]:
                    ans += f"{uriToFileName(uri)}:\n\t[Line {diagnostic.startLine+1}:{diagnostic.startCharacter+1}] {diagnostic.message}\n"
                ans += f"</new-diagnostics>"
    return ans
    

@dataclass
class SkillData:
    name: str
    description: str
    content: str
    fork: bool

def replacePlaceholders(line: str, arguments: list[str], skill: str) -> str:
    line = re.sub(r"`([^`]+/[^`]*)`", lambda m: f"`.claude/skills/{skill}/{m.group(1)}`",line)
    if not arguments:
        return line
    line = line.replace("$ARGUMENTS", " ".join(arguments))
    line = re.sub(r"\$ARGUMENT\[(\d+)\]", lambda m: str(arguments[int(m.group(1))]), line)
    line = re.sub(r"\$(\d+)", lambda m: str(arguments[int(m.group(1))]), line)
    return line

def buildSkill(skill_name: str, arguments: list[str]) -> SkillData:
    filePath = f".claude/skills/{skill_name}/SKILL.md"
    description = ""
    content = ""
    with open(filePath, 'r') as f:
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
            content += replacePlaceholders(line, arguments, skill_name)
        f.close()
    return SkillData(skill_name, description, content, fork)

def processSkills(skills: list[str], arguments: str) -> list[SkillData]:
    SKILLS = []
    for skill in skills:
        SKILLS.append(buildSkill(skill, arguments))
    return SKILLS

def getSkillsMessages(p: str) -> list[dict]:
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
    SKILLS = processSkills(skills, arguments)
    messages = [{"role": "system", "content": f"You have access to the following skills:\n\n- {"\n".join(f"{skill.name}: {skill.description}" for skill in SKILLS)}\n"}]
    for skill in SKILLS:
        messages.append({"role": "user", "content": skill.content})
    return messages

def getMessages(p: str) -> list[dict]:
    if p[0] == '/':
        return getSkillsMessages(p)
    else:
        messages = []
        skillsPath = Path(".claude/skills")
        if skillsPath.exists():
            skills = [skill.name for skill in skillsPath.iterdir() if skill.is_dir()]
            SKILLS = processSkills(skills, p)
            messages.append({"role": "system", "content": f"You have access to the following skills:\n\n{"\n".join(f"{skill.name}: {skill.description}" for skill in SKILLS)}\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."})
        messages.append({"role": "user", "content": p})
        return messages

def agentLoop(client: OpenAI, p: str):
    messages = getMessages(p)
    lspClient = LSPClient.from_environment()

    while True:
        diagnostics = ""
        if lspClient:
            diagnostics = getDiagnostics(lspClient)
            # print(diagnostics)
            if diagnostics != "":
                for message in reversed(messages):
                    if message["role"] in ("tool", "user"):
                        message["content"] += diagnostics
                        break
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

        diagnostics = ""
        wroteFile = False
        for tool_call in tool_calls:
            if lspClient and diagnostics == "":
                diagnostics = getDiagnostics(lspClient)
            if diagnostics != "":
                result = "Not executed, review diagnostics and retry"
            elif tool_call["name"] == "Write" and wroteFile:
                result = "Not executed; retry this write next turn"
            else:
                result = execute_tool(tool_call, client, lspClient)
                if tool_call == "Write":
                    wroteFile = True
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        if diagnostics != "":
            messages[-1]['content'] += diagnostics