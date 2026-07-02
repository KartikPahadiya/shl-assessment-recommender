import sys, importlib.util

spec = importlib.util.find_spec("app")
if spec:
    print(f"app module found at: {spec.origin}")
    print(f"Module path: {spec.submodule_search_locations}")
else:
    print("app module NOT found in sys.path")

spec2 = importlib.util.find_spec("app.main")
if spec2:
    print(f"app.main found at: {spec2.origin}")
else:
    print("app.main NOT found")

print(f"\nsys.path:")
for p in sys.path:
    print(f"  {p}")
