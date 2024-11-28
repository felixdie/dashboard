from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from helper_functions.backend_helpers import (
    initialise_llm,
    preprocess_data,
    initialise_RAG,
    create_retrival_chain,
    get_answer,
    get_logger,
    Master_Agent,
)
import chromadb
from config.ingest_config import config

# Create FastAPI instance
app = FastAPI()

# Allow input from frontend
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# chromadb.api.client.SharedSystemClient.clear_system_cache()

# Initialise logger
logger = get_logger()


class User_Input_Endpoint(BaseModel):
    user_input: str


# Endpoint to receive user input and return answer
@app.post("/user_input")
async def llm(
    User_Input_Endpoint: User_Input_Endpoint,
) -> str:

    user_input = User_Input_Endpoint.user_input

    logger.info(f"SUCCESS: User input received | {user_input}")

    # Initialise master agent
    master_agent = Master_Agent()

    # Determine task
    task = master_agent.choose_task(user_input=user_input)

    # Initialise LLM
    llm = initialise_llm(task=task)

    vectorstore = preprocess_data(task=task, user_input=user_input)

    query_transformer = initialise_RAG(
        vectorstore=vectorstore,
        llm=llm,
        task=task,
    )
    logger.info("SUCCESS: RAG initialised")

    retrival_chain = create_retrival_chain(
        llm=llm, query_transformer=query_transformer, task=task
    )
    logger.info("SUCCESS: Retrival Chain initialised")

    # Get answer
    answer = get_answer(query=user_input, query_chain=retrival_chain)
    logger.info("SUCCESS: Answer generated")

    # Reset vectorstore
    vectorstore.delete_collection()
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    logger.info("SUCCESS: Vectorstore cleared")

    # return answer to frontend
    logger.info(f"SUCCESS: Response returned | {answer}")
    return answer
