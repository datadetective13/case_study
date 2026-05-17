#!/usr/bin/env python3
import aws_cdk as cdk
from infrastructure.stacks.networking_stack import NetworkingStack
from infrastructure.stacks.data_stack import DataStack
from infrastructure.stacks.application_stack import ApplicationStack

app = cdk.App()

env = cdk.Environment(account="315311531132", region="us-east-1")

networking = NetworkingStack(app, "AWSDocsBotNetworking", env=env)
data = DataStack(app, "AWSDocsBotData", env=env)
application = ApplicationStack(
    app, "AWSDocsBotApplication",
    vpc=networking.vpc,
    opensearch_collection_endpoint=data.collection_endpoint,
    opensearch_collection_arn=data.collection_arn,
    docs_bucket_name=data.docs_bucket.bucket_name,
    conversation_table_name=data.conversation_table.table_name,
    env=env,
)
application.add_dependency(networking)
application.add_dependency(data)

app.synth()
