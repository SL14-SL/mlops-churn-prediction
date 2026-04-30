import json

def load_feature_schema(path="models/feature_schema.json"):
    with open(path, "r") as f:
        return json.load(f)
    
    