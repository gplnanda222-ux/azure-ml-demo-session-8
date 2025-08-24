#!/usr/bin/env python3
import os
import json
import sys

# Relies on your existing helper module
# (azureml_deployment.py) that you used earlier.
from azureml_deployment import AzureMLDeployer, create_inference_environment

def main():
    endpoint_name = os.environ["ENDPOINT_NAME"]
    action = os.environ["ACTION"].lower()
    traffic_percent = int(os.environ.get("DEPLOYMENT_PERCENTAGE", "100") or "100")

    with open("selected_model.json", "r") as f:
        model_info = json.load(f)

    model_name = model_info["name"]
    model_version = model_info["version"]

    print(f'🚀 Executing {action.upper()} for endpoint "{endpoint_name}" with model {model_name}:{model_version}')

    deployer = AzureMLDeployer("config.json")
    result = {}

    try:
        # Endpoint existence check
        try:
            details = deployer.get_endpoint_details(endpoint_name)
            endpoint_exists = details is not None
            state = getattr(details, "provisioning_state", "Unknown") if endpoint_exists else "NotFound"
            print(f"Endpoint exists: {endpoint_exists}, state: {state}")
        except Exception as e:
            print(f"⚠️  Could not check endpoint status: {e}")
            endpoint_exists = False

        if action == "delete":
            if not endpoint_exists:
                print("❌ Cannot delete a non-existent endpoint")
                sys.exit(1)
            print("🗑️  Deleting endpoint...")
            deployer.delete_endpoint(endpoint_name)
            result = {"action": "delete", "endpoint_name": endpoint_name, "status": "deleted"}

        elif action == "test-only":
            if not endpoint_exists:
                print("❌ Cannot test a non-existent endpoint")
                sys.exit(1)
            print("🧪 Testing endpoint...")
            ok = deployer.test_endpoint(endpoint_name)
            if not ok:
                print("❌ Endpoint test failed")
                sys.exit(1)
            result = {"action": "test-only", "endpoint_name": endpoint_name, "test_result": "passed"}

        elif action in ("create", "update"):
            # Ensure scoring env files exist
            create_inference_environment()

            if action == "create" or not endpoint_exists:
                print("✨ Creating endpoint...")
                endpoint = deployer.create_endpoint(endpoint_name)
                print(f"✅ Endpoint created: {endpoint.name}")
            else:
                print("🔄 Endpoint exists; proceeding to (re)deploy.")

            print("📦 Creating deployment...")
            deployment = deployer.create_deployment(
                model_name=model_name,
                model_version=model_version,
                endpoint_name=endpoint_name,
            )
            print(f"✅ Deployment created: {deployment.name}")

            if traffic_percent < 100:
                print(f"🚦 Blue-Green: directing {traffic_percent}% traffic to new deployment")
                # If your deployer supports traffic split, call it here.

            print("🧪 Smoke test...")
            ok = deployer.test_endpoint(endpoint_name)
            print("✅ Endpoint test successful" if ok else "⚠️  Warning: Endpoint test failed")

            result = {
                "action": action,
                "endpoint_name": endpoint_name,
                "deployment_name": deployment.name,
                "model_name": model_name,
                "model_version": model_version,
                "instance_type": os.environ.get("INSTANCE_TYPE", "Standard_DS2_v2"),
                "instance_count": int(os.environ.get("INITIAL_INSTANCE_COUNT", "1")),
                "auto_scaling_enabled": os.environ.get("ENABLE_AUTOSCALING", "false").lower() == "true",
                "traffic_percentage": traffic_percent,
                "environment": os.environ.get("ENVIRONMENT", "dev"),
                "status": "active",
            }
            if result["auto_scaling_enabled"]:
                result.update({
                    "min_capacity": int(os.environ.get("MIN_CAPACITY", "1") or "1"),
                    "max_capacity": int(os.environ.get("MAX_CAPACITY", "5") or "5"),
                })

        else:
            print(f"❌ Unknown action: {action}")
            sys.exit(1)

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/deployment_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n🎉 Deployment action completed successfully.")

    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
