import skore, json, pathlib

cfg = json.loads(pathlib.Path(".skore").read_text())
print("Current .skore config:")
for k, v in cfg.items():
    if k != "api_key":
        print(f"  {k}: {v}")

print()
skore.login(mode="hub")
print("Login OK")
