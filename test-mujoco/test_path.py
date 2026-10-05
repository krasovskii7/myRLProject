import os
from pathlib import Path

print("\n" + "<" + "="*50 + ">")
print(f"Script location: {Path(__file__).resolve()}")
print(f"Current Directory: {Path.cwd()}")
print("<" + "="*50 + ">")

print("Checking MJCF folder")
print(f"Script location: {Path(__file__).resolve().parents[1]}")
print(f"Script location: {Path(__file__).resolve().parents[1] / "DummyRobot"}")

path = Path(__file__).resolve().parents[1] / "DummyRobot"

for item in path.rglob("*"):
    if item.is_file():
        print(item)

