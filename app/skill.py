from dataclasses import dataclass
import re

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
        if l == "context: fork":
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



        