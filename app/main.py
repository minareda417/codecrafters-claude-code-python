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

def _replace_placeholders(line: str, arguments: list[str]) -> str:
    line = re.sub("$ARGUMENTS", " ".join(arguments), line)
    line = re.sub(r"\$ARGUMENT\[\d+\]", lambda m: str(arguments[int(m.group(1))]), line)
    line = re.sub(r"\$\d+", lambda m: str(arguments[int(m.group(1))]), line)
    return line

def get_skill_content(file_path: str, arguments: list[str]) -> str:
    content = ""
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
            content += _replace_placeholders(line, arguments)
        f.close()
    return content

def agent_loop(client: OpenAI, p):
    if p[0] == '/':
        command = p.split()
        skill = command[0]
        file_path = ".claude/skills/"+skill+"/SKILL.md"
        if len(p) > 1:
            arguments = p[1:]
        else:
            arguments = None
        messages = [{"role": "system", "content": f"You have access to the following skills:\n\n- {skill[1:]}: ..."},
                    {"role": "user", "content": get_skill_content(file_path, arguments)}]
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
