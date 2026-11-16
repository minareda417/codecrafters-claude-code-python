from pathlib import Path
from openai import OpenAI

from .skill import get_skills_messages, process_skills
from tool_executor import execute_tool, TOOLS

def get_messages(p: str) -> list[dict]:
    messages = []
    if p[0] == '/':
        messages.append(get_skills_messages(p))
    else:
        skills_path = Path(".claude/skills")
        if skills_path.exists():
            skills = [skill.name for skill in skills_path.iterdir() if skill.is_dir()]
            SKILLS = process_skills(skills, p)
            messages.append({"role": "system", "content": f"You have access to the following skills:\n\n{"\n".join(f"{skill.name}: {skill.description}" for skill in SKILLS)}\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."})
        messages.append({"role": "user", "content": p})
    return messages

def agent_loop(client: OpenAI, p: str):
    messages = get_messages(p)

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