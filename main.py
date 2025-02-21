# main.py
import asyncio
import logging

from api import load_apis
from agents import (
    APIRecommendationAgent,
    DocumentationClerkAgent,
    CodeAgent,
    CodeReviewAgent,
)
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Load Targeted API endpoints from CSV.
    PATH_TO_API = "..."
    api_df = load_apis(PATH_TO_API)
    logging.info("Loaded %d API entries.", len(api_df))
    
    valid_api_names = [row["API Name"] for _, row in api_df.iterrows()]
    api_list = [f"{row['API Name']}: {row['Description']}" for _, row in api_df.iterrows()]
    
    # Define a user query.
    user_query = (
        "I want to develop a Payment System, which has both physical and virtual terminal to serve different business needs. "
    )
    
    # Set up the model client.
    model_client = OpenAIChatCompletionClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434/v1",
        api_key="placeholder",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )
    
    # Instantiate agents.
    api_recommendation_agent = APIRecommendationAgent(
        model_client=model_client,
        valid_api_names=valid_api_names,
        api_list=api_list,
    )
    
    doc_clerk_agent = DocumentationClerkAgent(api_df=api_df, k=8)
    code_agent = CodeAgent(model_client=model_client)
    code_review_agent = CodeReviewAgent(model_client=model_client)
    
    termination = MaxMessageTermination(5)
    group_chat = RoundRobinGroupChat(
        participants=[
            api_recommendation_agent,
            doc_clerk_agent,
            code_agent,
            code_review_agent,
        ],
        termination_condition=termination,
    )
    
    # Run the group chat workflow.
    return Console(group_chat.run_stream(task=user_query))

if __name__ == "__main__":
    asyncio.run(main())