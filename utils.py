# utils.py
import re
import time
import logging
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from bs4 import BeautifulSoup

# For scraping
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)

def clean_api_names(raw_api_str: str, valid_api_names: List[str]) -> List[str]:
    """
    Clean and filter a comma-separated string of API names.
    
    Args:
        raw_api_str: A comma-separated string of API names.
        valid_api_names: A list of valid API names.
        
    Returns:
        A list of cleaned and filtered API names.
    """
    cleaned_str = re.sub(r'\s+', ' ', raw_api_str)
    candidates = [cand.strip() for cand in cleaned_str.split(',') if cand.strip()]
    valid_lookup = {name.lower(): name for name in valid_api_names}
    extracted = []
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in valid_lookup:
            extracted.append(valid_lookup[candidate_lower])
    return list(dict.fromkeys(extracted))

def extract_api_details(driver, doc_link: str, relevant_api: str) -> Dict[str, str]:
    """
    Extract API details (parameters, sortBy, cURL and JSON samples) from a documentation page.
    
    Args:
        driver: A Selenium WebDriver instance.
        doc_link: The URL of the documentation page.
        relevant_api: The name of the relevant API.
        
    Returns:
        A dictionary containing the extracted API details.
    """
    driver.get(doc_link)
    time.sleep(5)  # Wait for dynamic content to load
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    
    target_span = soup.find(
        'span',
        class_='sc-fzomuh eaYntv documentation-core-item-request-name',
        string=relevant_api
    )
    if not target_span:
        logging.warning("Target API '%s' not found on doc page %s", relevant_api, doc_link)
        return {}
    
    section = target_span.find_parent('section', class_='entitystyles__RequestContainer-sc-kfteh-7 hhPQzw')
    if not section:
        logging.warning("Details section not found for API '%s'", relevant_api)
        return {}
    
    details: Dict[str, str] = {
        "parameters": "",
        "sortby": "",
        "curl_input_sample": "",
        "json_output_sample": ""
    }
    
    info_div = section.find('div', class_='sc-fzpans jvNafo')
    if info_div:
        parameters_list = []
        sortby_list = []
        current_section = None
        all_spans = info_div.find_all('span')
        for sp in all_spans:
            text_val = sp.get_text(strip=True)
            text_lower = text_val.lower()
            if text_lower == "parameters":
                current_section = "parameters"
                continue
            elif text_lower == "sortby":
                current_section = "sortby"
                continue
            if current_section == "parameters":
                parameters_list.append(text_val)
            elif current_section == "sortby":
                sortby_list.append(text_val)
        details["parameters"] = "\n".join(parameters_list)
        details["sortby"] = "\n".join(sortby_list)
    
    curl_block = section.find('code', class_='language-curl highlighted-code__code')
    json_block = section.find('code', class_='language-json highlighted-code__code')
    if curl_block:
        details["curl_input_sample"] = curl_block.get_text(separator="\n", strip=True)
    if json_block:
        details["json_output_sample"] = json_block.get_text(separator="\n", strip=True)
    
    return details

def build_faiss_index(api_details: Dict[str, Dict[str, str]]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Build a Faiss index from API documentation details.
    
    Args:
        api_details: A dictionary mapping API names to their details.
        
    Returns:
        A tuple containing the built Faiss index and the corpus of API documentation chunks.
    """
    corpus = []
    for api_name, details in api_details.items():
        description = details.get('description') or ""
        parameters = details.get('parameters') or ""
        sortby = details.get('sortby') or ""
        curl_sample = details.get('curl_input_sample') or ""
        json_sample = details.get('json_output_sample') or ""
        text = (
            f"API: {api_name}\n"
            f"Description: {description}\n"
            f"Parameters: {parameters}\n"
            f"SortBy: {sortby}\n"
            f"cURL Sample: {curl_sample}\n"
            f"JSON Sample: {json_sample}"
        )
        corpus.append(text)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = embedding_model.encode(corpus, convert_to_numpy=True)
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(corpus_embeddings)
    logging.info("Faiss index built with %d entries.", index.ntotal)
    return index, corpus

def retrieve_context(query: str, index: faiss.IndexFlatL2, corpus: List[str], k: int = 2) -> List[str]:
    """
    Retrieve the most relevant API documentation chunks using the Faiss index.
    
    Args:
        query: The user's query.
        index: The built Faiss index.
        corpus: The corpus of API documentation chunks.
        k: The number of relevant chunks to retrieve (default is 2).
        
    Returns:
        A list of the most relevant API documentation chunks.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    contexts = [corpus[i] for i in indices[0] if i < len(corpus)]
    return contexts

def refine_documentation_context(contexts: List[str]) -> str:
    """
    Clean and join multiple documentation context strings.
    """
    joined_text = " ".join(contexts)
    cleaned_text = re.sub(r'\s+', ' ', joined_text)
    return cleaned_text