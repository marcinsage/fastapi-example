from langchain.agents.tools import Tool
import os
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import load_tools
from langchain.utilities import TextRequestsWrapper
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains import LLMChain
from langchain.agents.mrkl.output_parser import MRKLOutputParser
import json
from langchain.chat_models import ChatOpenAI
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
import autogen
from autogen import UserProxyAgent, AssistantAgent, config_list_from_models
import openai

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = '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'

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

service_account_file = "utils/service-account-key.json"
project = "source-398315"
dataset = "facebook_ads"
llm = ChatOpenAI(temperature=0, verbose=True, model="gpt-4")
sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_path={service_account_file}'
db = SQLDatabase.from_uri(sqlalchemy_url, include_tables=['ad_report'], sample_rows_in_table_info=2, view_support=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

def generate_llm_config(tool):
    function_schema = {
        "name": tool.name.lower().replace (' ', '_'),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    if tool.args is not None:
      function_schema["parameters"]["properties"] = tool.args

    return function_schema

tools = toolkit.get_tools()
function_map = {}

# Define the function
def sql_get_similar_examples(question):
    requests_tools = load_tools(["requests_all"])
    requests_wrapper = requests_tools[0].requests_wrapper
    headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer vYr6Jf03FSE4muGW3tZlibYS33rkbHvMmkHErL9GM89zCx1qt5ulvJI-BZWY",
                "Host": "app.airops.com"
            }
    url= "https://app.airops.com/public_api/airops_apps/503a9697-b2ab-458a-9f9b-bd8aa00d2f63/execute"
    payload = { "inputs": { "question": question } }
    requests = TextRequestsWrapper(headers=headers, aiosession=None)
    response= json.loads(requests.post(url=url, data=payload))['result']
    similiar_sql_queries = response  # Adjusted line to access the response with "similiar_sql_queries"
    print(similiar_sql_queries)  # Adjusted line to print "similiar_sql_queries"
    return similiar_sql_queries

# Add the function to the function_map
function_map['sql_get_similar_examples'] = sql_get_similar_examples

tool_names = ['InfoSQLDatabaseTool', 'ListSQLDatabaseTool', 'QuerySQLCheckerTool', 'QuerySQLDataBaseTool']
tool_schemas = []

for tool in tools:
    if tool.__class__.__name__ in tool_names:
        tool_schema = generate_llm_config(tool)
        tool_schemas.append(tool_schema)
        function_map[tool.name] = tool._run

# Add the function to the LLM config
llm_config = {
  "functions": tool_schemas + [{"name": "sql_get_similar_examples", "description": "Use this tool to retrieve similar SQL queries based on the user question that can be used as an example", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}}}],
  "config_list": config_list,
  "request_timeout": 120,
}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="""You are an agent designed to interact with a Google BigQuery SQL database. Your goal is to answer the questions from the user based on SQL queries that you perform.
User questions will be in the form of natural language questions, mostly related to the data for facebook ads marketing.
Given an input question, first retrieve similiar sql queries using the "sql_get_similiar_examples" tool with the user question as an input. If the example queries are enough to construct a query, then construct it. If not, use them as a reference and proceed by creating a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.

Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
Use only functions valid for BigQuery, qualify all field names, do not use correlated subqueries, use the GROUP BY statement when using aggregate functions (COUNT(), MAX(), MIN(), SUM(), AVG()).
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

Make sure to fully understand the user question and create a sql query that makes sense and represents the user question.
If the question does not seem related to the database or can not be answered due missing data, just return "I don't know" as the answer.
You have access to a tool which allows you to retrieve some similar previous questions and their correct SQL queries.
Reply "TERMINATE" in the end when everything is done.
IMPORTANT: ALWAYS construct your final answer using as a json string format: '"final_answer": "<your_answer>"
""",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE")
)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    function_map=function_map,
)


def initiate_chat_with_sql_agent(message):
    user_proxy.initiate_chat(
        chatbot,
        message=message,
        llm_config=llm_config,
    )
    last_message_from_chatbot = chatbot.last_message()["content"]
    if last_message_from_chatbot.endswith("TERMINATE"):
        last_message_from_chatbot = last_message_from_chatbot.replace("TERMINATE", "").strip()
    return last_message_from_chatbot

#test_question = "What is the average cost per purchase?"
#print(sql_get_similar_examples(test_question))
tools = [
    Tool(
        name="initiate_chat_with_sql_agent",
        func=lambda message: initiate_chat_with_sql_agent(message),
        description="Use this tool to initiate a a response from the sql agent for a given user question. Important: Input needs to be the user question itself or a semantically similiar request as an input.",
    )
]


prefix = """"

You're an helpful assistant that is answering user questions on data analytics for facebook ads. 
You have access to tools for interacting with the database.
Your goal is to answer the questions from the user using the provided tools.
If the user asks a question in german, then also answer in german using a conversational, non-formal tone of voice.
Except providing the final answer in german, do not use any other language than english.
"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")
tool_names = [tool.name for tool in tools]

output_parser = MRKLOutputParser()
llm_chain = LLMChain(llm = ChatOpenAI(temperature=0, verbose=True, model="gpt-4"), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, output_parser=output_parser)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)


def sql_chain(input):
    final_answer = agent_chain.run(input=input)
    return final_answer