import json
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_opensearchserverless as oss,
)
from constructs import Construct


class DataStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # S3 bucket for raw AWS documentation
        self.docs_bucket = s3.Bucket(
            self, "DocsBucket",
            bucket_name=f"aws-docs-bot-raw-{self.account}",
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # DynamoDB for conversation history
        self.conversation_table = dynamodb.Table(
            self, "ConversationTable",
            table_name="aws-docs-bot-conversations",
            partition_key=dynamodb.Attribute(
                name="session_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            time_to_live_attribute="ttl",
        )

        # OpenSearch Serverless — encryption policy
        encryption_policy = oss.CfnSecurityPolicy(
            self, "OssEncryptionPolicy",
            name="aws-docs-bot-enc",
            type="encryption",
            policy=json.dumps({
                "Rules": [{"ResourceType": "collection", "Resource": ["collection/aws-docs-bot"]}],
                "AWSOwnedKey": True,
            }),
        )

        # Network policy — public access (required for Lambda ingestion from VPC)
        network_policy = oss.CfnSecurityPolicy(
            self, "OssNetworkPolicy",
            name="aws-docs-bot-net",
            type="network",
            policy=json.dumps([{
                "Rules": [
                    {"ResourceType": "collection", "Resource": ["collection/aws-docs-bot"]},
                    {"ResourceType": "dashboard", "Resource": ["collection/aws-docs-bot"]},
                ],
                "AllowFromPublic": True,
            }]),
        )

        # OpenSearch Serverless collection
        self.collection = oss.CfnCollection(
            self, "OssCollection",
            name="aws-docs-bot",
            type="VECTORSEARCH",
            description="Vector store for AWS documentation embeddings",
        )
        self.collection.add_dependency(encryption_policy)
        self.collection.add_dependency(network_policy)

        # Data access policy — grants the account root full access (locked to task roles post-deploy)
        oss.CfnAccessPolicy(
            self, "OssDataAccessPolicy",
            name="aws-docs-bot-access",
            type="data",
            policy=json.dumps([{
                "Rules": [
                    {
                        "ResourceType": "index",
                        "Resource": ["index/aws-docs-bot/*"],
                        "Permission": [
                            "aoss:CreateIndex", "aoss:DeleteIndex",
                            "aoss:DescribeIndex", "aoss:ReadDocument",
                            "aoss:WriteDocument", "aoss:UpdateIndex",
                        ],
                    },
                    {
                        "ResourceType": "collection",
                        "Resource": ["collection/aws-docs-bot"],
                        "Permission": ["aoss:CreateCollectionItems", "aoss:DescribeCollectionItems", "aoss:UpdateCollectionItems"],
                    },
                ],
                "Principal": [f"arn:aws:iam::{self.account}:root"],
            }]),
        )

        self.collection_endpoint = self.collection.attr_collection_endpoint
        self.collection_arn = self.collection.attr_arn
