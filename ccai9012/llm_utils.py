"""
Large Language Model (LLM) Utilities Module
===============================

This module provides a comprehensive set of utilities for working with Large Language Models (LLMs),
particularly focused on the DeepSeek API. It includes functions for API key management,
model initialization, basic prompting, and advanced use cases such as structured output extraction
and retrieval-augmented generation.

Main components:
- API key management: Secure handling of DeepSeek API keys
- Basic LLM interaction: Functions to initialize models and send prompts
- Structured output utilities: Functions to extract structured data from reviews
- Retrieval utilities: Tools for document loading, chunking, embedding, and question answering

This module aims to simplify common LLM workflows and provide consistent interfaces for
various natural language processing tasks.

Usage:
    ### Basic usage
    llm = initialize_llm()
    ask_llm("What is the capital of France?", llm)
    
    ### Structured output extraction
    analyze_reviews(reviews_data, llm, "output.csv")
    
    ### Retrieval-based question answering
    retriever = build_pdf_retriever("document.pdf")
    run_qa_chain("What does the document say about climate change?", retriever, llm)
"""

import os
import getpass
from typing import Optional
from langchain_deepseek import ChatDeepSeek

import json
import time
from tqdm import tqdm
import pandas as pd


# ==========================
# Basic utilities
# ==========================

def get_deepseek_api_key(env_var: str = "DEEPSEEK_API_KEY") -> str:
    """
    Ensure the DeepSeek API key is set, or prompt the user to input it securely.

    This function first checks if the API key is available in the specified environment
    variable. If not found, it prompts the user to input it securely (without showing the
    keystrokes), and then sets it as an environment variable for future use.

    Parameters:
        env_var (str, optional): The name of the environment variable to check for the API key.
                               Defaults to "DEEPSEEK_API_KEY".

    Returns:
        str: The DeepSeek API key.

    Example:
        >>> api_key = get_deepseek_api_key()
        >>> # Or specify a custom environment variable
        >>> api_key = get_deepseek_api_key("MY_CUSTOM_DEEPSEEK_KEY")
    """
    api_key = os.getenv(env_var)
    if not api_key:
        api_key = getpass.getpass(f"Enter your {env_var}: ")
        os.environ[env_var] = api_key
    return api_key


def initialize_llm(
    model: str = "deepseek-chat",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    timeout: int = 30,
    max_retries: int = 3,
    api_key: Optional[str] = None
) -> ChatDeepSeek:
    """
    Initialize and configure a DeepSeek Chat model with the specified parameters.

    This function creates a ChatDeepSeek instance with the provided configuration,
    handling API key management automatically if not explicitly provided.

    Parameters:
        model (str, optional): The model identifier to use. Defaults to "deepseek-chat".
        temperature (float, optional): Controls randomness in generation. Higher values (e.g., 0.8)
                                     make output more random, lower values (e.g., 0.2) make it more
                                     deterministic. Defaults to 0.5.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 2048.
        timeout (int, optional): API request timeout in seconds. Defaults to 30.
        max_retries (int, optional): Number of times to retry failed API calls. Defaults to 3.
        api_key (str, optional): DeepSeek API key. If None, will be obtained using get_deepseek_api_key().

    Returns:
        ChatDeepSeek: Initialized DeepSeek language model client.

    Example:
        >>> # Basic initialization with default parameters
        >>> llm = initialize_llm()
        >>>
        >>> # Custom configuration
        >>> llm = initialize_llm(
        >>>     model="deepseek-chat",
        >>>     temperature=0.7,
        >>>     max_tokens=4096
        >>> )
    """
    if not api_key:
        api_key = get_deepseek_api_key()
    
    return ChatDeepSeek(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )


def ask_llm(prompt: str, llm: Optional[ChatDeepSeek] = None):
    """
    Send a prompt to the language model and stream the response to the console.

    This function provides a simple interface for interacting with a DeepSeek model.
    It prints both the input prompt and the generated response in a user-friendly format.
    If no LLM instance is provided, a default one will be initialized automatically.

    Parameters:
        prompt (str): The text prompt to send to the language model.
        llm (ChatDeepSeek, optional): An initialized DeepSeek language model client.
                                    If None, a default model will be initialized.

    Returns:
        None: The response is printed to the console.

    Example:
        >>> ask_llm("Explain quantum computing in simple terms.")
        >>>
        >>> # With custom LLM instance
        >>> custom_llm = initialize_llm(temperature=0.7)
        >>> ask_llm("Write a short poem about AI.", custom_llm)
    """
    if llm is None:
        llm = initialize_llm()

    print(f"\n📌 Prompt:\n{prompt}\n")
    for chunk in llm.stream(prompt):
        print(chunk.text(), end="")
    print("\n")


def generate_multiple_outputs(
    prompt: str,
    num_outputs: int = 3,
    temperature: float = 1.0,
    llm_params: Optional[dict] = None
):
    """
    Generate multiple responses to the same prompt, each with identical parameters.

    This function is useful for exploring the variety of responses that can be generated
    from a single prompt. It initializes a language model with the specified temperature
    and other parameters, then generates multiple responses sequentially.

    Parameters:
        prompt (str): The text prompt to send to the language model.
        num_outputs (int, optional): Number of different outputs to generate. Defaults to 3.
        temperature (float, optional): Temperature setting for generation. Higher values
                                      produce more diverse outputs. Defaults to 1.0.
        llm_params (dict, optional): Additional parameters to pass to initialize_llm().
                                   Defaults to {}.

    Returns:
        None: The responses are printed to the console.

    Example:
        >>> # Generate 3 different outputs with high creativity
        >>> generate_multiple_outputs(
        >>>     "Write a short marketing slogan for an AI product.",
        >>>     num_outputs=3,
        >>>     temperature=1.2
        >>> )
        >>>
        >>> # Generate more deterministic outputs with custom model settings
        >>> generate_multiple_outputs(
        >>>     "List 5 facts about machine learning.",
        >>>     num_outputs=2,
        >>>     temperature=0.3,
        >>>     llm_params={"max_tokens": 200, "model": "deepseek-chat"}
        >>> )
    """
    llm_params = llm_params or {}
    llm = initialize_llm(temperature=temperature, **llm_params)

    for i in range(num_outputs):
        print(f"\n Output #{i + 1} — Temperature {temperature}")
        print(f"📌 Prompt:\n{prompt}\n")
        for chunk in llm.stream(prompt):
            print(chunk.text(), end="")
        print("\n")

# ==========================
# Structure output utilities
# ==========================

def analyze_airbnb_reviews(reviews_df, llm, output_csv, max_reviews=50, sleep_time=0.1):
    """
    Analyze Airbnb reviews using a language model to extract structured opinion data.

    This function processes a dataset of Airbnb reviews, using a language model to extract
    structured information including overall impression, useful tags for potential guests,
    and opinions about specific aspects (location, facilities, host). Results are saved
    to a CSV file for further analysis.

    Parameters:
        reviews_df (pd.DataFrame): DataFrame containing Airbnb reviews with at least the columns
                                 'comments', 'listing_id', 'id', and 'date'.
        llm (ChatDeepSeek): An initialized DeepSeek language model client.
        output_csv (str): Path where the resulting CSV file will be saved.
        max_reviews (int, optional): Maximum number of reviews to process. Defaults to 50.
                                   A random sample is taken if the dataset is larger.
        sleep_time (float, optional): Delay in seconds between API calls to avoid rate limits.
                                    Defaults to 0.1.

    Returns:
        pd.DataFrame: The processed results as a DataFrame.

    Example:
        >>> reviews = pd.read_csv("airbnb_reviews.csv")
        >>> llm = initialize_llm()
        >>> results_df = analyze_airbnb_reviews(
        >>>     reviews,
        >>>     llm,
        >>>     "analyzed_reviews.csv",
        >>>     max_reviews=100
        >>> )
    """
    reviews_df = reviews_df.sample(n=max_reviews, random_state=42).reset_index(drop=True)
    results = []

    for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
        review_text = str(row['comments']).replace('\n', ' ').strip()

        prompt = f"""
You are analyzing Airbnb guest reviews to extract structured information.

For each review, extract the following:

1. overall_impression: One of "positive", "negative", or "neutral".
2. decision_tags: Generate 2 to 5 helpful tags for future guests (e.g., clean room, near MTR, great host).
3. highlighted_aspects: For each of these aspects – location, facility, host – extract:
   - aspect (one of "location", "facility", or "host")
   - opinion: "positive", "negative", or "neutral"
   - comment: A short sentence describing the opinion

Review: "{review_text}"

Return your answer strictly in the following JSON format:
{{
  "overall_impression": "...",
  "decision_tags": ["...", "..."],
  "highlighted_aspects": [
    {{
      "aspect": "location",
      "opinion": "...",
      "comment": "..."
    }},
    {{
      "aspect": "facility",
      "opinion": "...",
      "comment": "..."
    }},
    {{
      "aspect": "host",
      "opinion": "...",
      "comment": "..."
    }}
  ]
}}
        """.strip()

        try:
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += chunk.text()

            json_start = full_response.find('{')
            json_end = full_response.rfind('}') + 1
            json_str = full_response[json_start:json_end]

            data = json.loads(json_str)

            result = {
                "listing_id": row.get("listing_id"),
                "review_id": row.get("id"),
                "date": row.get("date"),
                "review_text": review_text,
                "overall_impression": data.get("overall_impression"),
                "decision_tags": ", ".join(data.get("decision_tags", []))
            }

            # Add aspects individually
            for aspect_data in data.get("highlighted_aspects", []):
                aspect = aspect_data.get("aspect")
                result[f"{aspect}_opinion"] = aspect_data.get("opinion")
                result[f"{aspect}_comment"] = aspect_data.get("comment")

            results.append(result)

        except Exception as e:
            print(f"\nError processing review: {e}")
            continue

        time.sleep(sleep_time)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} analyzed reviews to {output_csv}")
    return df


def load_business_locations(business_file):
    """
    Load all business entries with latitude and longitude from the Yelp dataset.

    This function reads a Yelp business.json file line by line, extracting businesses
    that have valid geographic coordinates. It's useful for applications requiring
    location-based analysis or visualization of business data.

    Parameters:
        business_file (str): Path to the Yelp business.json file.

    Returns:
        list: A list of dictionaries, where each dictionary contains business information:
              - name: Business name
              - latitude: Geographic latitude
              - longitude: Geographic longitude
              - city: City name (may be empty)
              - state: State code (may be empty)
              - categories: Business categories (may be empty)

    Example:
        >>> businesses = load_business_locations("data/yelp_reviews/business.json")
        >>> print(f"Loaded {len(businesses)} businesses with geographic coordinates")
        >>> # Plot businesses on a map
        >>> import folium
        >>> m = folium.Map(location=[businesses[0]['latitude'], businesses[0]['longitude']])
        >>> for business in businesses[:100]:  # Plot first 100 businesses
        >>>     folium.Marker(
        >>>         [business['latitude'], business['longitude']],
        >>>         tooltip=business['name']
        >>>     ).add_to(m)
    """
    businesses = []
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading businesses"):
            biz = json.loads(line)
            if biz.get('latitude') is not None and biz.get('longitude') is not None:
                businesses.append({
                    'name': biz['name'],
                    'latitude': biz['latitude'],
                    'longitude': biz['longitude'],
                    'city': biz.get('city', ''),
                    'state': biz.get('state', ''),
                    'categories': biz.get('categories', '')
                })
    return businesses


def load_reviews_by_city(business_file, review_file, city_name, max_reviews=1000):
    """
    Load Yelp reviews for businesses located in a specific city.

    This function performs a two-step process:
    1. First, it filters businesses located in the specified city
    2. Then, it loads reviews for those businesses, with additional location metadata

    This is especially useful for city-specific sentiment analysis and geographic
    visualization of review data.

    Parameters:
        business_file (str): Path to the Yelp business.json file.
        review_file (str): Path to the Yelp review.json file.
        city_name (str): Target city name (case-insensitive matching).
        max_reviews (int, optional): Maximum number of reviews to return. Defaults to 1000.

    Returns:
        list: A list of review dictionaries enriched with business location data:
             - business_id: Unique business identifier
             - business_name: Name of the business
             - text: Review text content
             - stars: Review rating (1-5)
             - date: Review date
             - latitude: Business geographic latitude
             - longitude: Business geographic longitude

    Example:
        >>> reviews = load_reviews_by_city(
        >>>     "data/yelp_reviews/business.json",
        >>>     "data/yelp_reviews/review.json",
        >>>     "Las Vegas",
        >>>     max_reviews=500
        >>> )
        >>> print(f"Loaded {len(reviews)} reviews from Las Vegas businesses")
    """
    business_ids_in_city = {}

    # First, load businesses in the target city
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            biz = json.loads(line)
            if biz.get('city', '').strip().lower() == city_name.strip().lower():
                business_ids_in_city[biz['business_id']] = {
                    'name': biz['name'],
                    'latitude': biz.get('latitude'),
                    'longitude': biz.get('longitude'),
                    'categories': biz.get('categories', ''),
                    'city': biz.get('city', '')
                }

    # Then, load reviews for those businesses
    reviews = []
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            biz_id = review['business_id']
            if biz_id in business_ids_in_city:
                # Combine review with business info
                review_data = {
                    'business_id': biz_id,
                    'business_name': business_ids_in_city[biz_id]['name'],
                    'text': review['text'],
                    'stars': review['stars'],
                    'date': review['date'],
                    'latitude': business_ids_in_city[biz_id]['latitude'],
                    'longitude': business_ids_in_city[biz_id]['longitude']
                }
                reviews.append(review_data)
                if len(reviews) >= max_reviews:
                    break

    return reviews


def analyze_reviews(reviews, llm, output_csv, max_reviews=50, sleep_time=0.2):
    """
    Analyze Yelp reviews using LLM to extract sentiment polarity, emotion, and keywords.

    This function processes Yelp reviews through a language model to extract structured
    information about sentiment and content. Each review is analyzed to determine:
    - Overall polarity (positive/negative/neutral)
    - Specific emotional sentiment (e.g., delighted, disappointed)
    - Key topics or aspects mentioned in the review

    Results are saved to a CSV file for further analysis or visualization.

    Parameters:
        reviews (list): List of review dictionaries, each containing at least a 'text' key.
                      May also contain metadata like 'stars', 'business_name', etc.
        llm (ChatDeepSeek): An initialized DeepSeek language model client.
        output_csv (str): Path where the resulting CSV file will be saved.
        max_reviews (int, optional): Maximum number of reviews to process. Defaults to 50.
        sleep_time (float, optional): Delay in seconds between API calls to avoid rate limits.
                                    Defaults to 0.2.

    Returns:
        pd.DataFrame: The processed results as a DataFrame with columns for review text,
                    polarity, sentiment, keywords, and any metadata from the original reviews.

    Example:
        >>> # First load reviews from a city
        >>> reviews = load_reviews_by_city(
        >>>     "data/yelp_reviews/business.json",
        >>>     "data/yelp_reviews/review.json",
        >>>     "San Francisco",
        >>>     max_reviews=200
        >>> )
        >>> # Then analyze them
        >>> llm = initialize_llm()
        >>> results_df = analyze_reviews(reviews, llm, "sf_review_analysis.csv", max_reviews=100)
        >>>
        >>> # Visualize results
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(10, 6))
        >>> results_df['polarity'].value_counts().plot(kind='bar')
        >>> plt.title('Sentiment Distribution in San Francisco Reviews')
        >>> plt.ylabel('Number of Reviews')
    """
    reviews = list(reviews)[:max_reviews]
    results = []

    for review in tqdm(reviews, total=len(reviews)):
        review_text = review['text'].replace('\n', ' ').strip()

        prompt = f"""
You are an assistant performing structured sentiment analysis on Yelp reviews.

Please analyze the following review and extract:

1. polarity: One of "positive", "negative", or "neutral".
2. sentiment: Choose the most appropriate emotion from the list: "delighted", "content", "surprise", "indifferent", "disappointed", "angry", or "frustrated".
3. keywords: Extract 2 to 5 important keywords that describe the main topics or aspects mentioned (e.g., cleanliness, service, location, waiting time).

Review: "{review_text}"

Return your answer strictly in the following JSON format:
{{
  "polarity": "...",
  "sentiment": "...",
  "keywords": ["...", "...", "..."]
}}
        """.strip()

        try:
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += chunk.text()

            # find JSON and extract
            json_start = full_response.find('{')
            json_end = full_response.rfind('}') + 1
            json_str = full_response[json_start:json_end]

            data = json.loads(json_str)

            results.append({
                "review_text": review_text,
                "polarity": data.get("polarity"),
                "sentiment": data.get("sentiment"),
                "keywords": ", ".join(data.get("keywords", [])),
                "stars": review.get("stars"),
                "business_name": review.get("business_name"),
                "latitude": review.get("latitude"),
                "longitude": review.get("longitude")
            })

        except Exception as e:
            print(f"\n Error processing review: {e}")
            continue

        time.sleep(sleep_time)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n Saved {len(df)} analyzed reviews to {output_csv}")
    return df


# ==========================
# Retrieval-related utilities
# ==========================
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
import os
import re

def build_pdf_retriever(
    pdf_path: str,
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    top_k: int = 10,
    exclude_last_n_pages: int = 2
):
    """
    Build a retriever for semantic search within PDF documents.

    This function implements a complete pipeline for PDF-based retrieval:
    1. Loading the PDF document and extracting text
    2. Chunking text into manageable segments with appropriate overlap
    3. Embedding text chunks using a vector embedding model
    4. Creating a FAISS vector store for efficient similarity search
    5. Returning a retriever that can find relevant document sections based on queries

    Parameters:
        pdf_path (str): Path to the PDF document to process.
        embedding_model_name (str, optional): Name of the HuggingFace embedding model to use.
                                           Defaults to "BAAI/bge-base-en-v1.5".
        chunk_size (int, optional): Maximum number of characters to process in a single chunk.
                                  Defaults to 1500.
        chunk_overlap (int, optional): Number of characters to overlap between consecutive chunks.
                                     Defaults to 200.
        top_k (int, optional): Number of top similar chunks to return for each query.
                             Defaults to 10.
        exclude_last_n_pages (int, optional): Number of pages to exclude from the end of the document
                                           (useful for skipping references, bibliographies).
                                           Defaults to 2.

    Returns:
        retriever: A LangChain retriever object ready for semantic search operations.

    Example:
        >>> # Build a retriever for a PDF document
        >>> retriever = build_pdf_retriever(
        >>>     pdf_path="data/energy-action-plan/1527001.pdf",
        >>>     chunk_size=2000,  # Larger chunks for more context
        >>>     top_k=5  # Return top 5 most relevant chunks
        >>> )
        >>>
        >>> # Use it with a language model for question answering
        >>> llm = initialize_llm()
        >>> run_qa_chain(
        >>>     "What are the main goals of the energy plan?",
        >>>     retriever,
        >>>     llm
        >>> )
    """
   # Load PDF pages as documents
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split each page into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(pages)

    # Clean metadata: keep only 'page' and 'source'
    for doc in docs:
        doc.metadata = {
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source")
        }

    # Exclude the last N pages (e.g., references)
    if exclude_last_n_pages > 0:
        max_page = max(doc.metadata.get("page", 0) for doc in docs)
        docs = [doc for doc in docs if doc.metadata.get("page", 0) <= max_page - exclude_last_n_pages]

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Build FAISS vector store from document chunks
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Create retriever with similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    return retriever


def run_qa_chain(
    query: str,
    retriever,
    llm,
    prompt_template: PromptTemplate = None,
    return_sources: bool = False,
    save_path: str = None,
):
    """
    Run a retrieval-based question answering chain on a given query.

    This function combines a retriever (which finds relevant document chunks) with a
    language model to generate answers based on the retrieved context. It provides
    options for customizing the prompt, viewing source documents, and saving results.

    Parameters:
        query (str): The question to be answered by the system.
        retriever: A LangChain retriever object (e.g., from build_pdf_retriever).
        llm: An initialized language model instance (e.g., from initialize_llm).
        prompt_template (PromptTemplate, optional): Custom prompt template to override the default.
                                                 Useful for specialized instructions or formatting.
        return_sources (bool, optional): Whether to print the source documents used for the answer.
                                      Defaults to False.
        save_path (str, optional): File path to save the result as a .txt file.
                                 If None, results are not saved. Defaults to None.

    Returns:
        str: The generated answer to the query based on retrieved documents.
    """
    chain_kwargs = {}
    if prompt_template is not None:
        chain_kwargs["prompt"] = prompt_template

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs=chain_kwargs,
        return_source_documents=return_sources,
    )

    response = qa_chain.invoke(query)

    print("\n--- Final Answer ---")
    print(response["result"])

    if return_sources:
        for i, doc in enumerate(response["source_documents"]):
            print(f"\n-------------------- Document {i+1} --------------------")
            print(doc.page_content)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response["result"])
        print(f"\n Answer saved to: {save_path}")

    return response["result"]


def parse_markdown_table(md_text: str) -> pd.DataFrame:
    """
    Convert a markdown-formatted table string into a pandas DataFrame.

    This function extracts and parses a markdown table from text input, handling special
    formatting like <br> tags in cells.

    Parameters:
        md_text (str): Text containing a markdown table (must include table rows starting with '|')

    Returns:
        pd.DataFrame: A pandas DataFrame containing the parsed table data

    Raises:
        ValueError: If no valid markdown table is found in the input text

    Example:
        >>> markdown = '''
        >>> | Name | Age | Occupation |
        >>> | ---- | --- | ---------- |
        >>> | John | 32  | Engineer   |
        >>> | Mary | 28  | Designer<br>Consultant |
        >>> '''
        >>> df = parse_markdown_table(markdown)
        >>> print(df.columns)  # ['Name', 'Age', 'Occupation']
        >>> print(df.loc[1, 'Occupation'])  # 'Designer; Consultant'
    """
    # Extract markdown table (start from the first '|')
    table_lines = [line for line in md_text.splitlines() if line.strip().startswith("|")]

    if len(table_lines) < 2:
        raise ValueError("No valid markdown table found.")

    # Extract header and rows
    header_line = table_lines[0]
    column_names = [col.strip() for col in header_line.strip().strip('|').split('|')]

    # Parse rows
    data_rows = []
    for line in table_lines[2:]:  # skip header and separator
        cells = [re.sub(r'<br\s*/?>', '; ', cell.strip(), flags=re.IGNORECASE) for cell in line.strip().strip('|').split('|')]
        # If row is shorter than columns, pad
        while len(cells) < len(column_names):
            cells.append("")
        data_rows.append(cells)

    df = pd.DataFrame(data_rows, columns=column_names)
    return df
