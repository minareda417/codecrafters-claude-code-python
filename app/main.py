import argparse
import os
import sys
import json
import subprocess
import re

from openai import OpenAI

from .tool_executor import tools, execute_tool

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

def _replace_placeholders(line: str, arguments: list[str], skill: str) -> str:
    line = re.sub(r"`([^`]+/[^`]*)`", lambda m: f"`.claude/skills/{skill}/{m.group(1)}`",line)
    if not arguments:
        return line
    line = line.replace("$ARGUMENTS", " ".join(arguments))
    line = re.sub(r"\$ARGUMENT\[(\d+)\]", lambda m: str(arguments[int(m.group(1))]), line)
    line = re.sub(r"\$(\d+)", lambda m: str(arguments[int(m.group(1))]), line)
    return line

def _get_skill_content(file_path: str, arguments: list[str], skill: str) -> str:
    content = ""
    description = ""
    with open(file_path, 'r') as f:
        counter = 0
        while counter < 2:
            line = f.readline()
            if line == "---\n":
                counter += 1
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            content += _replace_placeholders(line, arguments, skill)
        f.close()
    return content

def _get_description(file_path: str):
    description = ""
    with open(file_path, 'r') as f:
        for _ in range(2):
            f.readline()
        description += f.readline()[13:]
        f.close()
    return description

def _get_description_and_content(file_path: str, arguments: str, skill: str):
    content = ""
    description = ""
    with open(file_path, 'r') as f:
        for _ in range(2):
            f.readline()
        description += f.readline()[13:]
        f.readline()
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            content += _replace_placeholders(line, arguments, skill)
        f.close()
    return description, content

def _process_skills(skills: list[str], arguments: str) -> dict:
    skills_dict = {}
    for skill in skills:
        file_path = ".claude/skills/"+skill+"SKILL.md"
        skills_dict[skill]["description"], skills_dict[skill]["content"] = _get_description_and_content(file_path, arguments, skill)
    return skills_dict

def _get_skills_messages(p: str) -> list[dict]:
    command = p.split()
    skills = []
    idx = 0
    while idx < len(command) and command[idx][0] == '/' :
        skills.append(command[idx][1:])
        idx += 1
    if idx <= len(command) - 1:
        arguments = command[idx:]
    else:
        arguments = None
    skills_dict = _process_skills(skills, arguments)
    messages = [{"role": "system", "content": f"You have access to the following skills:\n\n- {"\n\n".join(f"{skill}: {skills_dict[skill]["description"]}" for skill in skills)}\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."}]
    print(messages)
    for skill in skills:
        messages.append({"role": "user", "content": skills_dict[skill]["content"]})
    return messages

def agent_loop(client: OpenAI, p: str):
    if p[0] == '/':
        messages = _get_skills_messages(p)
    else:
        messages=[{"role": "user", "content": p}]

    while True:
        response = client.chat.completions.create(model = "anthropic/claude-haiku-4.5",
                                                  messages = messages,
                                                  tools = tools)
        
        if not response.choices or len(response.choices) == 0:
                raise RuntimeError("no choices in response")
        
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
