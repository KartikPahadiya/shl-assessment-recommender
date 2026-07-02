import re

input_path = "app/data/catalog.json"
output_path = "app/data/catalog_fixed.json"

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace newlines inside quoted strings with spaces
def clean_json_string(match):
    content = match.group(0)
    return content.replace("\n", " ")

fixed = re.sub(r'"(?:[^"\\]|\\.)*"', clean_json_string, text, flags=re.DOTALL)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(fixed)

print("Saved fixed catalog")