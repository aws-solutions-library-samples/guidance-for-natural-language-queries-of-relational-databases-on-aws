# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and Amazon SageMaker JumpStart Foundation Models.
# Author: Gary A. Stafford (garystaf@amazon.com)
# Date: 2023-07-12
# Application expects the following environment variables (adjust for your environment):
# export ENDPOINT_NAME="hf-text2text-flan-t5-xxl-fp16"
# export REGION_NAME="us-east-1"
# Usage: streamlit run app.py --server.runOnSave true

import json
import logging
import os

import boto3
import streamlit as st
import yaml
from botocore.exceptions import ClientError
from langchain import (
    FewShotPromptTemplate,
    PromptTemplate,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from streamlit_chat import message

REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")


def main():
    st.set_page_config(
        page_title="Natural Language Query (NLQ) Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    NO_ANSWER_MSG = "I'm sorry, I was not able to answer your question."

    # Amazon SageMaker JumpStart Endpoint
    content_handler = ContentHandler()

    parameters = {
        "max_length": 2048,
        "temperature": 0.0,
    }

    llm = SagemakerEndpoint(
        endpoint_name=ENDPOINT_NAME,
        region_name=REGION_NAME,
        model_kwargs=parameters,
        content_handler=content_handler,
    )

    # define datasource uri
    rds_uri = get_rds_uri(REGION_NAME)
    db = SQLDatabase.from_uri(rds_uri)

    # load examples for few-shot prompting
    examples = load_samples()

    sql_db_chain = load_few_shot_chain(llm, db, examples)

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = []

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = []

    # define streamlit colums
    col1, col2 = st.columns([2, 1], gap="large")

    # build the streamlit sidebar
    build_sidebar()

    # build the main app ui
    build_form(col1, col2)

    # get the users query
    get_text(col1)
    user_input = st.session_state["query"]

    if user_input:
        with st.spinner(text="In progress..."):
            st.session_state.past.append(user_input)
            try:
                output = sql_db_chain(user_input)
                st.session_state.generated.append(output)
                logging.info(st.session_state["query"])
                logging.info(st.session_state["generated"])
            except Exception as exc:
                st.session_state.generated.append(NO_ANSWER_MSG)
                logging.error(exc)

    if st.session_state["generated"]:
        with col1:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                if (i >= 0) and (st.session_state["generated"][i] != NO_ANSWER_MSG):
                    message(
                        st.session_state["generated"][i]["result"],
                        key=str(i),
                        is_user=False,
                        avatar_style="icons",
                        seed="459",
                    )
                    message(
                        st.session_state["past"][i],
                        is_user=True,
                        key=str(i) + "_user",
                        avatar_style="icons",
                        seed="158",
                    )
                else:
                    message(
                        NO_ANSWER_MSG,
                        key=str(i),
                        is_user=False,
                        avatar_style="icons",
                        seed="459",
                    )
                    message(
                        st.session_state["past"][i],
                        is_user=True,
                        key=str(i) + "_user",
                        avatar_style="icons",
                        seed="158",
                    )

    position = len(st.session_state["generated"]) - 1
    with st.sidebar:
        if (position >= 0) and (
            st.session_state["generated"][position] != NO_ANSWER_MSG
        ):
            st.markdown("SageMaker JumpStart Foundation Model Endpoint:")
            st.code(ENDPOINT_NAME, language="text")
            st.markdown("Question:")
            st.code(st.session_state["generated"][position]["query"], language="text")
            st.markdown("SQL Query:")
            st.code(
                st.session_state["generated"][position]["intermediate_steps"][1],
                language="sql",
            )
            st.markdown("Results:")
            st.code(
                st.session_state["generated"][position]["intermediate_steps"][3],
                language="python",
            )
            st.markdown("Answer:")
            st.code(st.session_state["generated"][position]["result"], language="text")
        else:
            st.markdown("Nothing to see here...")


def get_rds_uri(region_name):
    # SQLAlchemy 2.0 reference: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
    # URI format: postgresql+psycopg2://user:pwd@hostname:port/dbname

    rds_username = None
    rds_password = None
    rds_endpoint = None
    rds_port = None
    rds_db_name = None

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        secret = client.get_secret_value(SecretId="/nlq/RDS_URI")
        secret = json.loads(secret["SecretString"])
        rds_endpoint = secret["RDSDBInstanceEndpointAddress"]
        rds_port = secret["RDSDBInstanceEndpointPort"]
        rds_db_name = secret["NLQAppDatabaseName"]

        secret = client.get_secret_value(SecretId="/nlq/NLQAppUsername")
        rds_username = secret["SecretString"]

        secret = client.get_secret_value(SecretId="/nlq/NLQAppUserPassword")
        rds_password = secret["SecretString"]
    except ClientError as e:
        logging.error(e)
        raise e

    return f"postgresql+psycopg2://{rds_username}:{rds_password}@{rds_endpoint}:{rds_port}/{rds_db_name}"


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]


def load_samples():
    # Use the corrected examples for few-shot prompting examples
    sql_samples = None

    with open("moma_examples.yaml", "r") as stream:
        sql_samples = yaml.safe_load(stream)

    return sql_samples


def load_few_shot_chain(llm, db, examples):
    example_prompt = PromptTemplate(
        input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
        template=(
            "{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult:"
            " {sql_result}\nAnswer: {answer}"
        ),
    )

    local_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        local_embeddings,
        Chroma,
        k=min(3, len(examples)),
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_postgres_prompt + "Here are some examples:",
        suffix=PROMPT_SUFFIX,
        input_variables=["table_info", "input", "top_k"],
    )

    return SQLDatabaseChain.from_llm(
        llm,
        db,
        prompt=few_shot_prompt,
        use_query_checker=True,  # must be True for flan-t5 models
        verbose=True,
        return_intermediate_steps=True,
    )


def get_text(col1):
    with col1:
        input_text = st.text_input(
            "Ask a question:",
            "",
            key="query_text",
            placeholder="Your question here...",
            on_change=clear_text(),
        )
        logging.info(input_text)


def clear_text():
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""


def build_sidebar():
    with st.sidebar:
        with st.container():
            st.markdown("# Under the Hood")


def build_form(col1, col2):
    with col1:
        with st.container():
            st.markdown("## Natural Language Query (NLQ) Demonstration")
            st.markdown(
                "Ask questions about The Museum of Modern Art (MoMA) Collection using natural language."
            )

        with st.container():
            with st.expander("Click here for sample questions..."):
                st.text(
                    """
                How many artists are there in the collection?
                How many pieces of artwork are there in the collection?
                How many artists are there whose nationality is Italian?
                How many artworks are by the artist Claude Monet?
                How many artworks are classified as paintings?
                How many artworks were created by Spanish artists?
                How many artist names start with the letter 'M'?
                ---
                How many artists are deceased as a percentage of all artists?
                Who is the most prolific artist in the collection? What is their nationality?
                What nationality of artists created the most artworks in the collection?
                What is the ratio of male to female artists? Return as a ratio.
                How many artworks were produced during the First World War, which are classified as paintings?
                What are the five oldest artworks in the collection? Return the title and date for each.
                Return the artwork for Frida Kahlo in a numbered list, including the title and date.
                What are the ten artworks by European artist, with a data? Write Python code to output them with Matplotlib as a table. Include a header row and font size of 12.
                ---
                Give me a recipe for chocolate cake.
                """
                )
    with col2:
        with st.container():
            st.button("clear chat", on_click=clear_session)


def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]


if __name__ == "__main__":
    main()
