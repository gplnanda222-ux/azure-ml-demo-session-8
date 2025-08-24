"""
Azure ML Model Registry management for loan approval model.
Handles model registration, versioning, tagging, and lifecycle management.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

class AzureMLModelRegistry:
    def __init__(self, config_path='config.json'):
        """Initialize Azure ML Model Registry client."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize ML client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.config['azure_ml']['subscription_id'],
            resource_group_name=self.config['azure_ml']['resource_group'],
            workspace_name=self.config['azure_ml']['workspace_name']
        )
        
        print(f"Connected to Azure ML workspace: {self.config['azure_ml']['workspace_name']}")

    def register_model_from_local(self, model_path: str = 'models', model_name: Optional[str] = None) -> Model:
        """Register model from local training artifacts."""
        if model_name is None:
            model_name = self.config.get('experiment', {}).get('model_name', 'loan-approval-model')
        
        print(f"Registering model from local artifacts: {model_path}")
        
        # Check if local artifacts exist
        model_file = os.path.join(model_path, 'model.pkl')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Local model file not found: {model_file}")
        
        # Load model metrics if available
        metrics = {}
        metrics_file = os.path.join('artifacts', 'model_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        # Create model registration
        model = Model(
            name=model_name,
            path=model_path,
            description=f"Loan approval prediction model trained locally",
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                "training_mode": "local",
                "model_type": self.config.get('model_hyperparameters', {}).get('model_type', 'random_forest'),
                "framework": "scikit-learn", 
                "environment": self.config.get('environment', 'dev'),
                "feature_store_enabled": "true",
                "created_by": "github_actions",
                "created_at": datetime.utcnow().isoformat(),
                "version_description": "Model from local training",
                **{f"metric_{k}": str(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            },
            properties={
                "training_dataset": "loan-approval-data",
                "validation_performed": "true",
                "model_format": "scikit-learn-pickle",
                "inference_ready": "true",
                "local_training": "true"
            }
        )
        
        # Register model
        try:
            registered_model = self.ml_client.models.create_or_update(model)
            print(f"✅ Model registered successfully: {registered_model.name}:{registered_model.version}")
            
            # Save registration info
            self._save_model_info(registered_model, "local_training")
            
            return registered_model
            
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            raise

    def register_model_from_job(self, job_name: str, model_name: Optional[str] = None) -> Model:
        """Register model from completed training job."""
        if model_name is None:
            model_name = self.config.get('experiment', {}).get('model_name', 'loan-approval-model')
        
        print(f"Registering model from job: {job_name}")
        
        # Get job details
        try:
            job = self.ml_client.jobs.get(job_name)
            if job.status != 'Completed':
                raise ValueError(f"Job {job_name} has not completed successfully. Status: {job.status}")
        except Exception as e:
            print(f"Failed to get job details: {e}")
            raise
        
        # Create model registration
        model = Model(
            name=model_name,
            path=f"azureml://jobs/{job_name}/outputs/artifacts/model",
            description=f"Loan approval prediction model trained from job {job_name}",
            type=AssetTypes.MLFLOW_MODEL,
            tags={
                "training_job": job_name,
                "model_type": self.config.get('model_hyperparameters', {}).get('model_type', 'random_forest'),
                "framework": "scikit-learn",
                "environment": self.config.get('environment', 'dev'),
                "feature_store_enabled": "true",
                "created_by": "github_actions",
                "created_at": datetime.utcnow().isoformat(),
                "version_description": f"Model from training job {job_name}"
            },
            properties={
                "training_dataset": "loan-approval-data",
                "validation_performed": "true",
                "model_format": "mlflow",
                "inference_ready": "true"
            }
        )
        
        # Register model
        try:
            registered_model = self.ml_client.models.create_or_update(model)
            print(f"✅ Model registered successfully: {registered_model.name}:{registered_model.version}")
            
            # Save registration info
            self._save_model_info(registered_model, job_name)
            
            return registered_model
            
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            raise

    def _save_model_info(self, model: Model, job_name: str):
        """Save model registration information."""
        os.makedirs('artifacts', exist_ok=True)
        
        model_info = {
            "name": model.name,
            "version": model.version,
            "id": model.id,
            "training_job": job_name,
            "registration_time": datetime.utcnow().isoformat(),
            "tags": model.tags,
            "properties": model.properties,
            "description": model.description
        }
        
        with open('artifacts/registered_model.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model info saved to artifacts/registered_model.json")

    def list_models(self, model_name: Optional[str] = None) -> List[Model]:
        """List models in registry."""
        try:
            if model_name:
                models = list(self.ml_client.models.list(name=model_name))
                print(f"Found {len(models)} versions of model '{model_name}'")
            else:
                models = list(self.ml_client.models.list())
                print(f"Found {len(models)} models in registry")
            
            # Display model info
            for model in models:
                created_time = getattr(model, 'creation_context', {}).get('created_at', 'Unknown')
                print(f"  - {model.name}:{model.version} (Created: {created_time})")
                if hasattr(model, 'tags') and model.tags:
                    env = model.tags.get('environment', 'Unknown')
                    model_type = model.tags.get('model_type', 'Unknown')
                    print(f"    Environment: {env}, Type: {model_type}")
            
            return models
            
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []

    def add_model_tags(self, model_name: str, version: str, tags: Dict[str, str]):
        """Add tags to existing model."""
        try:
            # Get existing model
            model = self.ml_client.models.get(name=model_name, version=version)
            
            # Update tags
            existing_tags = model.tags or {}
            existing_tags.update(tags)
            
            # Create updated model
            updated_model = Model(
                name=model.name,
                version=model.version,
                path=model.path,
                description=model.description,
                type=model.type,
                tags=existing_tags,
                properties=model.properties
            )
            
            # Update model
            result = self.ml_client.models.create_or_update(updated_model)
            print(f"✅ Tags added to model {model_name}:{version}")
            for key, value in tags.items():
                print(f"  - {key}: {value}")
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to add tags to model: {e}")
            raise

def main():
    """Main model registry management function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure ML Model Registry Management')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--register-from-job', type=str, help='Register model from training job')
    parser.add_argument('--register-from-local', action='store_true', help='Register model from local artifacts')
    parser.add_argument('--model-name', type=str, help='Model name')
    parser.add_argument('--model-version', type=str, help='Model version')
    parser.add_argument('--list-models', action='store_true', help='List all models')
    parser.add_argument('--add-tags', action='store_true', help='Add tags to model')
    parser.add_argument('--tags', nargs='*', help='Tags in format key=value')
    parser.add_argument('--model-info', type=str, help='Model info in format name:version')
    
    args = parser.parse_args()
    
    # Initialize model registry
    registry = AzureMLModelRegistry(args.config)
    
    if args.register_from_job:
        model = registry.register_model_from_job(args.register_from_job, args.model_name)
        print(f"Model registered: {model.name}:{model.version}")
    
    if args.register_from_local:
        model = registry.register_model_from_local('models', args.model_name)
        print(f"Model registered: {model.name}:{model.version}")
    
    if args.list_models:
        models = registry.list_models(args.model_name)
    
    if args.add_tags:
        if args.model_info and args.tags:
            name, version = args.model_info.split(':')
            tags_dict = {}
            for tag in args.tags:
                if '=' in tag:
                    key, value = tag.split('=', 1)
                    tags_dict[key] = value
            registry.add_model_tags(name, version, tags_dict)
        else:
            print("Error: --model-info and --tags required for add-tags")

if __name__ == "__main__":
    main()