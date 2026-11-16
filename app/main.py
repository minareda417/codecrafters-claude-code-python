import argparse
import os
import sys
import json
import subprocess
import re
from pathlib import Path

from openai import OpenAI
from .skill import skill, build_skill

from .tool_executor import TOOLS, execute_tool

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def _process_skills(skills: list[str], arguments: str) -> list[skill]:
    SKILLS = []
    for skill in skills:
        SKILLS.append(build_skill(skill, arguments))
    return SKILLS

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
        arguments = []
    SKILLS = _process_skills(skills, arguments)
    messages = [{"role": "system", "content": f"You have access to the following skills:\n\n- {"\n".join(f"{skill.name}: {skill.description}" for skill in SKILLS)}\n"}]
    for skill in SKILLS:
        messages.append({"role": "user", "content": skill.content})
    return messages

def agent_loop(client: OpenAI, p: str):
    if p[0] == '/':
        messages = _get_skills_messages(p)
    else:
        skills_path = Path(".claude/skills")
        messages = []
        if skills_path.exists():
            skills = [skill.name for skill in skills_path.iterdir() if skill.is_dir()]
            SKILLS = _process_skills(skills, p)
            messages.append{"role": "system", "content": f"You have access to the following skills:\n\n{"\n".join(f"{skill.name}: {skill.description}" for skill in SKILLS)}\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."}
        messages.append({"role": "user", "content": p})

    while True:
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
