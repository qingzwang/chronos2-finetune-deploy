#!/usr/bin/env python3
"""
Deploy finetuned Chronos-2 model to Amazon SageMaker as a real-time endpoint.

This script:
1. Downloads the base JumpStart Chronos-2 model artifacts (inference code)
2. Replaces model weights with the finetuned checkpoint
3. Packages and uploads to S3
4. Creates a SageMaker endpoint

Usage:
    python deploy/deploy_to_sagemaker.py

    # Custom options
    python deploy/deploy_to_sagemaker.py \
        --model_path output/chronos2-336-48-covariant-full-100k-w-covariates-covariate_dropout-4-gpus/finetuned-ckpt \
        --endpoint_name chronos2-finetuned-zendure \
        --instance_type ml.g5.2xlarge \
        --region us-west-2

    # Delete endpoint when done
    python deploy/deploy_to_sagemaker.py --delete --endpoint_name chronos2-finetuned-zendure
"""

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.jumpstart.model import JumpStartModel


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy finetuned Chronos-2 to SageMaker")
    parser.add_argument(
        "--model_path", type=str,
        default="output/chronos2-336-48-covariant-full-100k-w-covariates-covariate_dropout-4-gpus/finetuned-ckpt",
        help="Local path to finetuned model checkpoint",
    )
    parser.add_argument("--endpoint_name", type=str, default="chronos2-finetuned-zendure")
    parser.add_argument("--instance_type", type=str, default="ml.g5.2xlarge")
    parser.add_argument("--initial_instance_count", type=int, default=1)
    parser.add_argument("--region", type=str, default=None,
                        help="AWS region (default: from session)")
    parser.add_argument("--s3_bucket", type=str, default=None,
                        help="S3 bucket for model artifacts (default: SageMaker default bucket)")
    parser.add_argument("--s3_prefix", type=str, default="chronos2-finetuned-zendure")
    parser.add_argument("--role", type=str, default=None,
                        help="SageMaker execution role ARN (default: auto-detect)")
    parser.add_argument("--delete", action="store_true",
                        help="Delete the endpoint instead of deploying")
    return parser.parse_args()


def download_jumpstart_artifacts(js_model, tmpdir):
    """Download JumpStart base model artifacts from S3."""
    s3 = boto3.client("s3")
    s3_source = js_model.model_data["S3DataSource"]
    s3_uri = s3_source["S3Uri"].rstrip("/") + "/"
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)

    print(f"Downloading JumpStart artifacts from {s3_uri} ...")
    artifact_dir = Path(tmpdir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = key[len(prefix):]
            local_file = artifact_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_file))
            count += 1

    print(f"  Downloaded {count} files to {artifact_dir}")
    return artifact_dir


def replace_model_weights(artifact_dir, finetuned_model_path):
    """Replace the base model weights with finetuned checkpoint."""
    finetuned_path = Path(finetuned_model_path)
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Finetuned model not found: {finetuned_path}")

    # Find where model weights are stored in the JumpStart artifacts
    # JumpStart puts model files directly in the artifact root
    print(f"Replacing model weights with finetuned checkpoint from {finetuned_path} ...")

    # Copy finetuned model files (config.json, model.safetensors, etc.)
    for src_file in finetuned_path.iterdir():
        if src_file.is_file():
            dst_file = artifact_dir / src_file.name
            print(f"  Copying {src_file.name} -> {dst_file}")
            shutil.copy2(src_file, dst_file)


def package_and_upload(artifact_dir, s3_bucket, s3_key):
    """Create tar.gz from artifacts and upload to S3."""
    print("Packaging model artifacts ...")
    tar_path = Path(artifact_dir).parent / "model.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        for item in Path(artifact_dir).iterdir():
            tar.add(str(item), arcname=item.name)

    tar_size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"  Archive size: {tar_size_mb:.1f} MB")

    print(f"Uploading to s3://{s3_bucket}/{s3_key} ...")
    s3 = boto3.client("s3")
    s3.upload_file(str(tar_path), s3_bucket, s3_key)

    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"  Uploaded to {s3_uri}")
    return s3_uri


def deploy_endpoint(model_data_uri, image_uri, role, endpoint_name, instance_type, instance_count, sagemaker_session):
    """Create SageMaker Model and deploy as real-time endpoint."""
    print(f"\nDeploying endpoint: {endpoint_name}")
    print(f"  Instance type: {instance_type}")
    print(f"  Instance count: {instance_count}")
    print(f"  Image URI: {image_uri}")

    model = Model(
        name=endpoint_name,
        model_data=model_data_uri,
        image_uri=image_uri,
        role=role,
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
    )

    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    print(f"\nEndpoint deployed successfully: {endpoint_name}")
    return predictor


def delete_endpoint(endpoint_name):
    """Delete a SageMaker endpoint and its resources."""
    sm = boto3.client("sagemaker")

    print(f"Deleting endpoint: {endpoint_name}")
    try:
        sm.delete_endpoint(EndpointName=endpoint_name)
        print("  Endpoint deleted.")
    except sm.exceptions.ClientError as e:
        print(f"  Endpoint delete error: {e}")

    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print("  Endpoint config deleted.")
    except sm.exceptions.ClientError:
        pass

    try:
        sm.delete_model(ModelName=endpoint_name)
        print("  Model deleted.")
    except sm.exceptions.ClientError:
        pass


def test_endpoint(endpoint_name, sagemaker_session):
    """Quick smoke test to verify the endpoint is working."""
    print(f"\nTesting endpoint: {endpoint_name}")

    predictor = Predictor(
        endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    payload = {
        "inputs": [
            {"target": [0.0, 4.0, 5.0, 1.5, -3.0, -5.0, -3.0, 1.5, 5.0, 4.0,
                         0.0, -4.0, -5.0, -1.5, 3.0, 5.0, 3.0, -1.5, -5.0, -4.0]},
        ],
        "parameters": {"prediction_length": 10},
    }

    response = predictor.predict(payload)
    predictions = response["predictions"][0]
    print(f"  Mean forecast: {[round(v, 2) for v in predictions['mean']]}")
    print("  Smoke test passed!")
    return response


def main():
    args = parse_args()

    if args.delete:
        delete_endpoint(args.endpoint_name)
        return

    # Setup session
    if args.region:
        boto_session = boto3.Session(region_name=args.region)
    else:
        boto_session = boto3.Session()
    session = sagemaker.Session(boto_session=boto_session)
    region = session.boto_region_name
    bucket = args.s3_bucket or session.default_bucket()
    role = args.role or sagemaker.get_execution_role(sagemaker_session=session)

    print("=" * 70)
    print("Deploy Finetuned Chronos-2 to SageMaker")
    print("=" * 70)
    print(f"  Model path:     {args.model_path}")
    print(f"  Endpoint name:  {args.endpoint_name}")
    print(f"  Instance type:  {args.instance_type}")
    print(f"  Region:         {region}")
    print(f"  S3 bucket:      {bucket}")
    print(f"  Role:           {role[:50]}...")
    print("=" * 70)

    # Step 1: Get JumpStart model info (image URI, base artifacts)
    print("\nStep 1: Getting JumpStart model configuration ...")
    js_model = JumpStartModel(
        model_id="pytorch-forecasting-chronos-2",
        instance_type=args.instance_type,
        role=role,
        region=region,
        sagemaker_session=session,
    )
    image_uri = js_model.image_uri
    print(f"  Image URI: {image_uri}")

    # Step 2: Download base artifacts, replace weights, package and upload
    print("\nStep 2: Preparing model artifacts ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = download_jumpstart_artifacts(js_model, tmpdir)
        replace_model_weights(artifact_dir, args.model_path)

        s3_key = f"{args.s3_prefix}/model.tar.gz"
        model_data_uri = package_and_upload(artifact_dir, bucket, s3_key)

    # Step 3: Deploy
    print("\nStep 3: Deploying endpoint ...")
    predictor = deploy_endpoint(
        model_data_uri=model_data_uri,
        image_uri=image_uri,
        role=role,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        instance_count=args.initial_instance_count,
        sagemaker_session=session,
    )

    # Step 4: Smoke test
    test_endpoint(args.endpoint_name, session)

    print(f"\nDone! Endpoint '{args.endpoint_name}' is ready.")
    print(f"To delete: python deploy/deploy_to_sagemaker.py --delete --endpoint_name {args.endpoint_name}")


if __name__ == "__main__":
    main()
