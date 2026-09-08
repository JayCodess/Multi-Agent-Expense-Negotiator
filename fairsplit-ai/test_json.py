import re
import json

raw = """```json
{
  "proposal": {
    "Aisha": 8500.0,
    "Raj": 15500.0,
    "Meera": 6000.0
  },
  "reasoning": "test"
}
```"""

cleaned = raw.strip()
match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
if match:
    cleaned = match.group(1).strip()
    
print("Cleaned:", repr(cleaned))
print("Parsed:", json.loads(cleaned))
