# agents.py
import time
import re
import logging
from typing import AsyncGenerator, Sequence, List, Dict

import asyncio

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core.models import AssistantMessage, UserMessage, RequestUsage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import ChatMessage, AgentEvent, TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.base import Response

# Import helper functions from utils
from utils import (
    clean_api_names,
    extract_api_details,
    build_faiss_index,
    retrieve_context,
    refine_documentation_context,
)

class APIRecommendationAgent(AssistantAgent):
    """
    APIRecommendationAgent receives a user query and a list of targeted APIs.
    It then recommends relevant APIs based on the user query and the targeted API list.
    """
    def __init__(
        self,
        model_client,
        name: str = "APIRecommendationAgent",
        valid_api_names: List[str] = None,
        api_list: List[str] = None,
    ):
        super().__init__(
            name=name,
            description="Agent that recommends relevant targeted APIs.",
            model_client=model_client,
        )
        self.model_client = model_client
        self.valid_api_names = valid_api_names or []
        self.api_list = api_list or []
        self._model_context = UnboundedChatCompletionContext()
        self._system_message = (
            "You are an API recommendation assistant for targeted.\n"
            "You receive a user query about building a software application.\n"
            "You reference a known API list to identify which targeted APIs are relevant.\n"
            "Return only a comma-separated list of matching API names, without any extra text."
        )
        
    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message

        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")

        return final_response

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[Response, None]:
        for msg in messages:
            role = getattr(msg, "role", None) or msg.source
            if role.lower() == "user":
                await self._model_context.add_message(UserMessage(content=msg.content, source=role))
            else:
                await self._model_context.add_message(AssistantMessage(content=msg.content, source=role))

        user_query = messages[-1].content if messages else ""
        prompt = f"""
Given the following API list:
{self.api_list}

Identify the API names that are helpful to the following user query:
"{user_query}"
Return only a comma-separated list of API names without any additional text.
"""

        response_message = await self.model_client.create([UserMessage(content=prompt, source="user")])
        raw_response = response_message.content.strip()
        logging.info("Raw LLM response: %s", raw_response)
        recommended_apis = clean_api_names(raw_response, self.valid_api_names)
        if recommended_apis:
            final_str = ", ".join(recommended_apis)
        else:
            final_str = "No relevant APIs found."
            logging.info("No matching APIs found; terminating agent response.")
            await self._model_context.add_message(AssistantMessage(content=final_str, source=self.name))
            yield Response(
                chat_message=TextMessage(content=final_str, source=self.name),
                inner_messages=[]
            )
            return
        
        logging.info("Cleaned Recommended API Names: %s", recommended_apis)
        await self._model_context.add_message(AssistantMessage(content=final_str, source=self.name))
        yield Response(
            chat_message=TextMessage(content=final_str, source=self.name),
            inner_messages=[]
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()


class DocumentationClerkAgent(BaseChatAgent):
    """
    A Documentation Clerk Agent that takes as input a message containing a comma-separated list of API names.
    It then looks up each API's documentation link from the provided CSV DataFrame,
    scrapes documentation details via Selenium, builds a Faiss index for additional context retrieval,
    and produces a final aggregated documentation summary.
    """
    def __init__(
        self,
        name: str = "DocumentationClerkAgent",
        api_df = None,
        k: int = 2,
    ):
        super().__init__(name=name, description="Agent that retrieves and summarizes API documentation details.")
        self.api_df = api_df  # A DataFrame with targeted API info.
        self.k = k
        self._model_context = UnboundedChatCompletionContext()
        self._system_message = (
            "You are a documentation clerk. Your job is to compile the documentation details for a list of APIs. "
            "For each API, scrape its documentation page to extract parameters, sortBy, cURL sample, and JSON sample. "
            "Then, build a combined summary that also retrieves additional context via a RAG pipeline. "
            "Return a concise, aggregated summary of all documentation details."
        )
        
    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message

        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")
        return final_response
    
    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        if not messages:
            yield Response(chat_message=TextMessage(content="No message provided.", source=self.name, models_usage=RequestUsage(prompt_tokens=0, completion_tokens=0)), inner_messages=[])
            return
        
        for msg in messages:
            await self._model_context.add_message(UserMessage(content=msg.content, source=msg.source))

        recommended_str = messages[-1].content
        logging.info("DocumentationClerkAgent received message: %s", recommended_str)
        relevant_api_names = [x.strip() for x in recommended_str.split(",") if x.strip()]
        logging.info("DocumentationClerkAgent received API names: %s", relevant_api_names)
        
        original_user_query = None
        for msg in messages:
            role = getattr(msg, "role", None) or msg.source
            if role.lower() == "user":
                original_user_query = msg.content
                break
        logging.info("Original user query: %s", original_user_query)
        
        all_api_details: Dict[str, Dict[str, str]] = {}
        
        # Set up Selenium driver once.
        logging.info("Setting up Selenium driver...")
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium import webdriver
        from webdriver_manager.chrome import ChromeDriverManager

        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        for api_name in relevant_api_names:
            logging.info("Extracting details for API: %s", api_name)
            row = self.api_df[self.api_df["API Name"] == api_name]
            if row.empty:
                logging.warning("API '%s' not found in CSV.", api_name)
                continue
            api_desc = row["Description"].values[0]
            api_url = row["API Endpoints"].values[0]
            doc_link = row["Document Link"].values[0]
            logging.info("Extracting details for API: %s from %s", api_name, doc_link)
            
            details = extract_api_details(driver, doc_link, api_name)
            all_api_details[api_name] = {
                "description": api_desc,
                "api_url": api_url,
                "doc_link": doc_link,
                "parameters": details.get("parameters", "No parameters provided."),
                "sortby": details.get("sortby", "No sortby information provided."),
                "curl_input_sample": details.get("curl_input_sample", "No cURL sample provided."),
                "json_output_sample": details.get("json_output_sample", "No JSON sample provided.")
            }
        
        driver.quit()
        
        logging.info("API details extracted for %d APIs.", len(all_api_details))
        if relevant_api_names:
            logging.info("Example of the first API detail:\n%s", all_api_details.get(relevant_api_names[0], {}))
        
        index, corpus = build_faiss_index(all_api_details)
        num_relevant_apis = min(self.k, len(relevant_api_names))
        retrieved = retrieve_context(original_user_query, index, corpus, k=num_relevant_apis)
        context_text = refine_documentation_context(retrieved)

        final_summary = f"You should integrate these {num_relevant_apis} most relevant APIs into the application.\n"
        final_summary += "Retrieved Context (from API documentation):\n"
        final_summary += "------------------------------------------------------\n"
        final_summary += context_text
        final_summary += "\n------------------------------------------------------\n\n"

        await self._model_context.add_message(AssistantMessage(content=final_summary, source=self.name))
        yield Response(
            chat_message=TextMessage(content=final_summary, source=self.name),
            inner_messages=[]
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()


class CodeAgent(AssistantAgent):
    """
    CodeAgent receives the aggregated API documentation summary (from DocumentationClerkAgent)
    and the original user query from the conversation history. It then constructs an integrated prompt
    to generate full-stack code (Python backend + HTML/JS frontend) and returns the generated code.
    """
    def __init__(
        self,
        model_client,
        name: str = "CodeAgent",
    ):
        system_prompt = (
            "You are a full-stack developer AI assistant. Based on the aggregated API documentation details "
            "and the user's requirement, generate a complete solution that includes:\n"
            "1. A Python backend using Flask (or FastAPI) that calls these APIs.\n"
            "2. A basic HTML/JavaScript frontend that communicates with the backend.\n"
            "Include necessary comments and provide only the code."
        )
        super().__init__(name=name, description="Agent that generates full-stack code.", model_client=model_client)
        self.model_client = model_client
        self._model_context = UnboundedChatCompletionContext()
        self._system_message = system_prompt
        
    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)    

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message
        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")
        return final_response

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[Response, None]:
        original_user_query = None
        aggregated_doc_summary = None
        for msg in messages:
            role = getattr(msg, "role", None) or msg.source
            if role.lower() == "user" and original_user_query is None:
                original_user_query = msg.content
            elif role.lower() == "documentationclerkagent":
                aggregated_doc_summary = msg.content

        if not original_user_query:
            original_user_query = "No user query provided."
        if not aggregated_doc_summary:
            aggregated_doc_summary = "No aggregated documentation provided."

        prompt = (
            "You are a full-stack developer AI assistant.\n"
            f"User Requirement: \"{original_user_query}\"\n\n"
            "Aggregated API Documentation:\n"
            "------------------------------------------------------\n"
            f"{aggregated_doc_summary}\n"
            "------------------------------------------------------\n\n"
            "Using the above information, generate a complete solution that includes:\n"
            "1. A Python backend using Flask (or FastAPI) that calls these APIs.\n"
            "2. A basic HTML/JavaScript frontend that communicates with the backend.\n"
            "Provide only the code, with necessary comments.\n"
        )
        response_message = await self.model_client.create([UserMessage(content=prompt, source="user")])
        generated_code = response_message.content.strip()
        await self._model_context.add_message(AssistantMessage(content=generated_code, source=self.name))
        yield Response(
            chat_message=TextMessage(content=generated_code, source=self.name),
            inner_messages=[]
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()


class CodeReviewAgent(AssistantAgent):
    """
    CodeReviewAgent reviews the full-stack code generated by CodeAgent.
    It extracts the code from the conversation, refines it to improve its structure,
    readability, and robustness, and then returns only the refined code along with a brief
    summary of the refinement logic.
    
    The expected output should consist of:
      - The refined code.
      - A section titled 'Refinement Summary:' describing the changes made.
    """
    def __init__(
        self,
        model_client,
        name: str = "CodeReviewAgent",
    ):
        system_prompt = (
            "You are a code reviewer. Your task is to review the full-stack code generated by CodeAgent, "
            "refine it to improve its structure, readability, and robustness, and then provide only the final "
            "refined code along with a brief summary of your refinement logic. "
            "Your output should contain the refined code followed by a section titled 'Refinement Summary:' which explains your changes."
        )
        super().__init__(name=name, description="Agent that reviews and refines generated full-stack code.", model_client=model_client)
        self.model_client = model_client
        self._model_context = UnboundedChatCompletionContext()
        self._system_message = system_prompt

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[Response, None]:
        generated_code = None
        for msg in messages:
            role = getattr(msg, "role", None) or msg.source
            if role.lower() == "codeagent":
                generated_code = msg.content
                break
        if not generated_code:
            generated_code = "No code generated."

        prompt = (
            "You are a code reviewer. Below is the full-stack code generated by CodeAgent:\n\n"
            f"{generated_code}\n\n"
            "Please refine the code to improve its structure, readability, and robustness. "
            "Then, provide only the final refined code along with a brief summary of your refinement logic. "
            "Your output should contain the refined code followed by a section titled 'Refinement Summary:' which explains your changes."
        )
        response_message = await self.model_client.create([UserMessage(content=prompt, source="user")])
        refined_output = response_message.content.strip()
        await self._model_context.add_message(AssistantMessage(content=refined_output, source=self.name))
        yield Response(
            chat_message=TextMessage(content=refined_output, source=self.name),
            inner_messages=[]
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()