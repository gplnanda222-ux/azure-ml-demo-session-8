#!/usr/bin/env python3
import os
import json
import sys

from azureml_model_registry import AzureMLModelRegistry

def main():
    model_name = os.environ["MODEL_NAME"]
    model_version = os.environ.get("MODEL_VERSION", "").strip()

    print("🔍 Validating deployment configuration...")
    print(f"Model: {model_name}")
    print(f"Version: {model_version or 'latest'}")
    print(f"Endpoint: {os.environ['ENDPOINT_NAME']}")
    print(f"Action: {os.environ['ACTION']}")
    print(f"Environment: {os.environ['ENVIRONMENT']}")
    print(f"Instance Type: {os.environ['INSTANCE_TYPE']}")
    print(f"Initial Count: {os.environ['INITIAL_INSTANCE_COUNT']}")
    print(f"Auto-scaling: {os.environ['ENABLE_AUTOSCALING']}")

    if os.environ["ENABLE_AUTOSCALING"].lower() == "true":
        print(f"Min Capacity: {os.environ.get('MIN_CAPACITY', '1')}")
        print(f"Max Capacity: {os.environ.get('MAX_CAPACITY', '5')}")

    try:
        registry = AzureMLModelRegistry("config.json")
        models = registry.list_models(os.environ["MODEL_NAME"])
        if not models:
            raise ValueError("No models found")

        # pick latest
        target_model = models[0]
        model_info = {
            "name": target_model.name,
            "version": target_model.version,
            "id": target_model.id,
            "tags": getattr(target_model, "tags", {}),
            "stage": getattr(target_model, "tags", {}).get("stage", "Unknown"),
        }
        with open("selected_model.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print("✅ Model validated", model_info)

    except Exception as e:
        print(f"❌ Error validating models: {e}")
        sys.exit(1)

