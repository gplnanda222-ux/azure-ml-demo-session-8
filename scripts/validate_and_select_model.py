import os, sys, json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

try:
    # Authenticate
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"]
    )

    model_name = os.environ["MODEL_NAME"]
    target_version = os.environ.get("MODEL_VERSION")

    print(f"🔍 Checking model: {model_name}, version: {target_version or 'latest'}")

    # List models
    models = list(ml_client.models.list(name=model_name))
    if not models:
        print(f"❌ No models found with name {model_name}")
        sys.exit(1)

    # Pick version
    if target_version:
        model = next((m for m in models if m.version == target_version), None)
        if not model:
            print(f"❌ Model {model_name} version {target_version} not found")
            versions = [m.version for m in models]
            print(f"Available versions: {versions}")
            sys.exit(1)
    else:
        # Latest is first when sorted by version number
        model = sorted(models, key=lambda m: int(m.version), reverse=True)[0]

    # Save info
    model_info = {
        "name": model.name,
        "version": model.version,
        "id": model.id,
        "tags": getattr(model, "tags", {}),
        "stage": getattr(model, "tags", {}).get("stage", "Unknown")
    }
    with open("selected_model.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"✅ Selected model {model.name}:{model.version}, stage={model_info['stage']}")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
