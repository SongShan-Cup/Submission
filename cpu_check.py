import json
from pathlib import Path

total = sum(i * i for i in range(32))
assert total == 10416
result = {"sum_squares": total, "message": "manual trial"}
Path("cpu_result.json").write_text(json.dumps(result), encoding="utf-8")
print("CPU_OK", total, flush=True)