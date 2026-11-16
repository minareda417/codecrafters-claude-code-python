from dataclasses import dataclass
import re

@dataclass
class skill:
    name: str
    description: str
    content: str

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
        description += f.readline[13:]
        for _ in range(2):
            f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            content += _replace_placeholders(line, arguments, skill_name)
        f.close()
    return skill(skill_name, description, content)



        