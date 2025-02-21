# Agentic AI System for API Integration

## Overview

This project implements an Agentic AI system designed to help developers quickly understand and integrate various APIs into their applications. The system guides the user through the entire development process—from API recommendation and documentation retrieval to full‑stack code generation and code review.

## Pipeline Architecture

The project is organized into four main Python scripts, each responsible for a distinct part of the workflow:

- **`api.py`**:  
  Contains the function to load API endpoint data from a CSV file.  
  **Role**: Data ingestion of API metadata.

- **`utils.py`**:  
  Provides helper functions including:
  - Cleaning and validating API names.
  - Scraping API documentation details using Selenium and BeautifulSoup.
  - Building and querying a Faiss index for retrieval-augmented generation (RAG).
  - Refining documentation context.
  
- **`agents.py`**:  
  Defines the core multi-agent components:
  - **APIRecommendationAgent**:  
    Processes the user query and recommends relevant APIs from the pre-loaded CSV.
  
  - **DocumentationClerkAgent**:  
    Takes the recommended API names, scrapes their documentation, and aggregates details (parameters, filtering options, request/response samples). It further leverages a Faiss index to retrieve additional context.
  
  - **CodeAgent**:  
    Generates full‑stack code (a Python backend using Flask or FastAPI and an HTML/JavaScript frontend) based on the aggregated documentation and the original user query.
  
  - **CodeReviewAgent**:  
    Reviews and refines the generated code, outputting production-ready code along with a brief summary of the improvements.

- **`main.py`**:  
  Integrates all modules by:
  - Loading the API data.
  - Instantiating the agents.
  - Setting up a round‑robin group chat session.
  - Executing the multi-agent workflow.
  
  **Usage**: Running `main.py` will execute the complete pipeline and display the final output in the console.

## Pipeline Execution Flow

1. **User Query**:  
   The process begins with a user query (e.g., *"I want to develop a Payment System that supports both physical and virtual transactions."*).

2. **API Recommendation**:  
   `APIRecommendationAgent` analyzes the query against a pre-loaded list of APIs and returns a comma-separated list of relevant API names.

3. **Documentation Aggregation**:  
   `DocumentationClerkAgent` uses Selenium and BeautifulSoup to scrape documentation for the recommended APIs. It then builds a Faiss index to retrieve the most relevant documentation context based on the original query and outputs a concise aggregated summary.

4. **Code Generation**:  
   `CodeAgent` combines the aggregated documentation with the user query to generate a complete full‑stack solution. This includes a Python backend (using Flask or FastAPI) and an HTML/JavaScript frontend.

5. **Code Review**:  
   `CodeReviewAgent` reviews and refines the generated code, ensuring improved structure, readability, and robustness. It outputs the final refined code along with a summary of the refinement logic.

6. **Termination**:  
   The conversation is automatically terminated based on a predefined condition (using `MaxMessageTermination`).

## How to Run

1. **Install Dependencies**:  
   Ensure that you have all the necessary Python packages installed. Key dependencies include:
   - `pandas`
   - `beautifulsoup4`
   - `selenium` and `webdriver_manager`
   - `faiss`
   - `sentence_transformers`
   - `autogen_agentchat` (and related AutoGen libraries)

2. **Set Up the Environment**:  
   - Place the API metadata CSV file in the `./data/` directory with the name `api_data.csv`.
   - Update the configuration in `main.py` (e.g., model client settings, file paths).

3. **Run the Pipeline**:  
   Execute the following command in your terminal:
   ```bash
   python main.py