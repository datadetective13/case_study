from aws_cdk import (
    Stack, Duration, Size,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_logs as logs,
)
from constructs import Construct
import os


BEDROCK_LLM_MODEL = "us.meta.llama3-3-70b-instruct-v1:0"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
OSS_INDEX_NAME = "aws-docs"


class ApplicationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.Vpc,
        opensearch_collection_endpoint: str,
        opensearch_collection_arn: str,
        docs_bucket_name: str,
        conversation_table_name: str,
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc, container_insights=True)

        # Shared IAM policy for Bedrock + OpenSearch + DynamoDB
        bedrock_policy = iam.ManagedPolicy(
            self, "BedrockPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                    resources=[
                        f"arn:aws:bedrock:{self.region}::foundation-model/{BEDROCK_LLM_MODEL}",
                        f"arn:aws:bedrock:{self.region}::{BEDROCK_LLM_MODEL}",  # inference profile ARN
                        f"arn:aws:bedrock:{self.region}::foundation-model/{BEDROCK_EMBED_MODEL}",
                    ],
                ),
                iam.PolicyStatement(
                    actions=["aoss:APIAccessAll"],
                    resources=[opensearch_collection_arn],
                ),
                iam.PolicyStatement(
                    actions=["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan"],
                    resources=[f"arn:aws:dynamodb:{self.region}:{self.account}:table/{conversation_table_name}"],
                ),
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:ListBucket"],
                    resources=[
                        f"arn:aws:s3:::{docs_bucket_name}",
                        f"arn:aws:s3:::{docs_bucket_name}/*",
                    ],
                ),
            ],
        )

        # ── Backend (FastAPI) ──────────────────────────────────────────────────
        backend_image = ecr_assets.DockerImageAsset(
            self, "BackendImage",
            directory=os.path.join(os.path.dirname(__file__), "../../backend"),
            platform=ecr_assets.Platform.LINUX_AMD64,
        )

        backend_task_role = iam.Role(
            self, "BackendTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[bedrock_policy],
        )

        backend_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "BackendService",
            cluster=cluster,
            cpu=1024,
            memory_limit_mib=2048,
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(backend_image),
                container_port=8000,
                task_role=backend_task_role,
                environment={
                    "OSS_ENDPOINT": opensearch_collection_endpoint,
                    "OSS_INDEX": OSS_INDEX_NAME,
                    "OSS_REGION": self.region,
                    "BEDROCK_LLM_MODEL": BEDROCK_LLM_MODEL,
                    "BEDROCK_EMBED_MODEL": BEDROCK_EMBED_MODEL,
                    "DYNAMO_TABLE": conversation_table_name,
                },
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="backend",
                    log_retention=logs.RetentionDays.ONE_WEEK,
                ),
            ),
            public_load_balancer=True,
            listener_port=80,
        )
        backend_service.target_group.configure_health_check(path="/health")

        backend_url = f"http://{backend_service.load_balancer.load_balancer_dns_name}"

        # ── Frontend (Streamlit) ───────────────────────────────────────────────
        frontend_image = ecr_assets.DockerImageAsset(
            self, "FrontendImage",
            directory=os.path.join(os.path.dirname(__file__), "../../frontend"),
            platform=ecr_assets.Platform.LINUX_AMD64,
        )

        frontend_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "FrontendService",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(frontend_image),
                container_port=8501,
                environment={
                    "BACKEND_URL": backend_url,
                },
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="frontend",
                    log_retention=logs.RetentionDays.ONE_WEEK,
                ),
            ),
            public_load_balancer=True,
            listener_port=80,
        )
        frontend_service.target_group.configure_health_check(path="/_stcore/health")

        # ── Ingestion Lambda ───────────────────────────────────────────────────
        ingestion_role = iam.Role(
            self, "IngestionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                bedrock_policy,
            ],
        )
        ingestion_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                resources=[
                    f"arn:aws:s3:::{docs_bucket_name}",
                    f"arn:aws:s3:::{docs_bucket_name}/*",
                ],
            )
        )

        ingestion_fn = lambda_.Function(
            self, "IngestionFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            # deps are pre-installed into the ingestion/ directory (no Docker bundling needed)
            code=lambda_.Code.from_asset(
                os.path.join(os.path.dirname(__file__), "../../ingestion"),
            ),
            role=ingestion_role,
            timeout=Duration.minutes(15),
            memory_size=1024,
            environment={
                "OSS_ENDPOINT": opensearch_collection_endpoint,
                "OSS_INDEX": OSS_INDEX_NAME,
                "OSS_REGION": self.region,
                "BEDROCK_EMBED_MODEL": BEDROCK_EMBED_MODEL,
                "DOCS_BUCKET": docs_bucket_name,
            },
        )

        # Output the public URLs
        from aws_cdk import CfnOutput
        CfnOutput(self, "FrontendURL", value=f"http://{frontend_service.load_balancer.load_balancer_dns_name}")
        CfnOutput(self, "BackendURL", value=backend_url)
        CfnOutput(self, "IngestionFunctionName", value=ingestion_fn.function_name)
