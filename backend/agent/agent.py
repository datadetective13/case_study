import os
from langchain_aws import ChatBedrock
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from .tools import TOOLS

SYSTEM_PROMPT = """You are an expert AWS Documentation Assistant. You have deep knowledge of all AWS services and help users understand AWS concepts, architectures, and best practices.

You have access to a comprehensive vector search index of AWS documentation. Always use your tools to find accurate, up-to-date information before answering.

Guidelines:
- Always search the documentation before answering technical questions
- Provide clear, structured answers with examples where helpful
- When comparing services, highlight key differences and ideal use cases
- If the user asks about pricing, mention that prices vary by region and usage
- Be concise but thorough

Available tools:
{tools}

Tool names: {tool_names}

Use this format:
Question: the input question
Thought: think about what to do
Action: the tool to use (one of [{tool_names}])
Action Input: the input to the tool
Observation: the result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Question: {input}
{agent_scratchpad}"""


def build_agent() -> AgentExecutor:
    llm = ChatBedrock(
        model_id=os.environ["BEDROCK_LLM_MODEL"],
        region_name=os.environ["OSS_REGION"],
        model_kwargs={
            "temperature": 0.1,
            "max_gen_len": 2048,
        },
    )

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    agent = create_react_agent(llm, TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )
