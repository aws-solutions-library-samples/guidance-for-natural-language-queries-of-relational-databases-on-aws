# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and Amazon Bedrock.
# Author: Gary A. Stafford (garystaf@amazon.com)
# Date: 2023-08-12
# Usage: streamlit run app_bedrock.py --server.runOnSave true

import ast
import json
import logging
import os

import boto3
import pandas as pd
import streamlit as st
import yaml
from botocore.exceptions import ClientError
from langchain.sql_database import SQLDatabase
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import Bedrock
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from langchain_experimental.sql import SQLDatabaseChain

# ***** CONFIGURABLE PARAMETERS *****
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")
MODEL_NAME = os.environ.get("MODEL_NAME", "anthropic.claude-instant-v1")
TEMPERATURE = os.environ.get("TEMPERATURE", 0.3)
MAX_TOKENS_TO_SAMPLE = os.environ.get("MAX_TOKENS_TO_SAMPLE", 4096)
TOP_K = os.environ.get("TOP_K", 250)
TOP_P = os.environ.get("TOP_P", 1)
STOP_SEQUENCES = os.environ.get("STOP_SEQUENCES", ["\n\nHuman"])
BASE_AVATAR_URL = (
    "https://raw.githubusercontent.com/garystafford-aws/static-assets/main/static"
)
ASSISTANT_ICON = os.environ.get("ASSISTANT_ICON", "bot-64px.png")
USER_ICON = os.environ.get("USER_ICON", "human-64px.png")
HUGGING_FACE_EMBEDDINGS_MODEL = os.environ.get(
    "HUGGING_FACE_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# ******************************************************************


def main():
    st.set_page_config(
        page_title="NLQ Demo",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # # hide the hamburger bar menu
    # hide_streamlit_style = """
    #     <style>
    #     #MainMenu {visibility: hidden;}
    #     footer {visibility: hidden;}
    #     </style>

    # """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    NO_ANSWER_MSG = "Sorry, I was unable to answer your question."

    parameters = {
        "max_tokens_to_sample": MAX_TOKENS_TO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "stop_sequences": STOP_SEQUENCES,
    }

    llm = Bedrock(
        region_name=REGION_NAME,
        model_id=MODEL_NAME,
        model_kwargs=parameters,
        verbose=True,
    )

    # define datasource uri
    rds_uri = get_rds_uri(REGION_NAME)
    db = SQLDatabase.from_uri(rds_uri)

    # load examples for few-shot prompting
    examples = load_samples()

    sql_db_chain = load_few_shot_chain(llm, db, examples)

    # store the initial value of widgets in session state
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

    tab1, tab2, tab3 = st.tabs(["Chatbot", "Details", "Technologies"])

    with tab1:
        col1, col2 = st.columns([6, 1], gap="medium")

        with col1:
            with st.container():
                st.markdown("## The Museum of Modern Art (MoMA) Collection")
                st.markdown(
                    "#### Query the collection’s dataset using natural language."
                )
                st.markdown(" ")
                with st.expander("Click here for sample questions..."):
                    st.markdown(
                        """
                        - Simple
                            - How many artists are there in the collection?
                            - How many pieces of artwork are there?
                            - How many artists are there whose nationality is Italian?
                            - How many artworks are by the artist Claude Monet?
                            - How many artworks are classified as paintings?
                            - How many artworks were created by Spanish artists?
                            - How many artist names start with the letter 'M'?
                        - Moderate
                            - How many artists are deceased as a percentage of all artists?
                            - Who is the most prolific artist? What is their nationality?
                            - What nationality of artists created the most artworks?
                            - What is the ratio of male to female artists? Return as a ratio.
                        - Complex
                            - How many artworks were produced during the First World War, which are classified as paintings?
                            - What are the five oldest pieces of artwork? Return the title and date for each.
                            - What are the 10 most prolific artists? Return their name and count of artwork.
                            - Return the artwork for Frida Kahlo in a numbered list, including the title and date.
                            - What is the count of artworks by classification? Return the first ten in descending order. Don't include Not_Assigned.
                            - What are the 12 artworks by different Western European artists born before 1900? Write Python code to output them with Matplotlib as a table. Include header row and font size of 12.
                        - Unrelated to the Dataset
                            - Give me a recipe for chocolate cake.
                            - Who won the 2022 FIFA World Cup final?
                    """
                    )
                st.markdown(" ")
            with st.container():
                input_text = st.text_input(
                    "Ask a question:",
                    "",
                    key="query_text",
                    placeholder="Your question here...",
                    on_change=clear_text(),
                )
                logging.info(input_text)

                user_input = st.session_state["query"]

                if user_input:
                    with st.spinner(text="Thinking..."):
                        st.session_state.past.append(user_input)
                        try:
                            output = sql_db_chain(user_input)
                            st.session_state.generated.append(output)
                            logging.info(st.session_state["query"])
                            logging.info(st.session_state["generated"])
                        except Exception as exc:
                            st.session_state.generated.append(NO_ANSWER_MSG)
                            logging.error(exc)

                # https://discuss.streamlit.io/t/streamlit-chat-avatars-not-working-on-cloud/46713/2
                if st.session_state["generated"]:
                    with col1:
                        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                            if (i >= 0) and (
                                st.session_state["generated"][i] != NO_ANSWER_MSG
                            ):
                                with st.chat_message(
                                    "assistant",
                                    avatar=f"{BASE_AVATAR_URL}/{ASSISTANT_ICON}",
                                ):
                                    st.write(st.session_state["generated"][i]["result"])
                                with st.chat_message(
                                    "user",
                                    avatar=f"{BASE_AVATAR_URL}/{USER_ICON}",
                                ):
                                    st.write(st.session_state["past"][i])
                            else:
                                with st.chat_message(
                                    "assistant",
                                    avatar=f"{BASE_AVATAR_URL}/{ASSISTANT_ICON}",
                                ):
                                    st.write(NO_ANSWER_MSG)
                                with st.chat_message(
                                    "user",
                                    avatar=f"{BASE_AVATAR_URL}/{USER_ICON}",
                                ):
                                    st.write(st.session_state["past"][i])
        with col2:
            with st.container():
                st.button("clear chat", on_click=clear_session)
    with tab2:
        with st.container():
            st.markdown("### Details")
            st.markdown("Amazon Bedrock Model:")
            st.code(MODEL_NAME, language="text")

            position = len(st.session_state["generated"]) - 1
            if (position >= 0) and (
                st.session_state["generated"][position] != NO_ANSWER_MSG
            ):
                st.markdown("Question:")
                st.code(
                    st.session_state["generated"][position]["query"], language="text"
                )

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
                st.code(
                    st.session_state["generated"][position]["result"], language="text"
                )

                data = ast.literal_eval(
                    st.session_state["generated"][position]["intermediate_steps"][3]
                )
                if len(data) > 0 and len(data[0]) > 1:
                    df = None
                    st.markdown("Pandas DataFrame:")
                    df = pd.DataFrame(data)
                    df
    with tab3:
        with st.container():
            st.markdown("### Technologies")
            st.markdown(" ")

            st.markdown("##### Natural Language Query (NLQ)")
            st.markdown(
                """
            [Natural language query (NLQ)](https://www.yellowfinbi.com/glossary/natural-language-query), according to Yellowfin, enables analytics users to ask questions of their data. It parses for keywords and generates relevant answers sourced from related databases, with results typically delivered as a report, chart or textual explanation that attempt to answer the query, and provide depth of understanding.
            """
            )
            st.markdown(" ")

            st.markdown("##### The MoMa Collection Datasets")
            st.markdown(
                """
            [The Museum of Modern Art (MoMA) Collection](https://github.com/MuseumofModernArt/collection) contains over 120,000 pieces of artwork and 15,000 artists. The datasets are available on GitHub in CSV format, encoded in UTF-8. The datasets are also available in JSON. The datasets are provided to the public domain using a [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/).
            """
            )
            st.markdown(" ")

            st.markdown(" ")

            st.markdown("##### Amazon Bedrock")
            st.markdown(
                """
            [Amazon Bedrock](https://aws.amazon.com/bedrock/) is the easiest way to build and scale generative AI applications with foundation models (FMs).
            """
            )

            st.markdown("##### LangChain")
            st.markdown(
                """
            [LangChain](https://python.langchain.com/en/latest/index.html) is a framework for developing applications powered by language models. LangChain provides standard, extendable interfaces and external integrations.
            """
            )
            st.markdown(" ")

            st.markdown("##### Chroma")
            st.markdown(
                """
            [Chroma](https://www.trychroma.com/) is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.
            """
            )
            st.markdown(" ")

            st.markdown("##### Streamlit")
            st.markdown(
                """
            [Streamlit](https://streamlit.io/) is an open-source app framework for Machine Learning and Data Science teams. Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
            """
            )

        with st.container():
            st.markdown("""---""")
            st.markdown(
                "![](app/static/github-24px-blk.png) [Feature request or bug report?](https://github.com/aws-solutions-library-samples/guidance-for-natural-language-queries-of-relational-databases-on-aws/issues)"
            )
            st.markdown(
                "![](app/static/github-24px-blk.png) [The MoMA Collection datasets on GitHub](https://github.com/MuseumofModernArt/collection)"
            )
            st.markdown(
                "![](app/static/flaticon-24px.png) [Icons courtesy flaticon](https://www.flaticon.com)"
            )


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


def load_samples():
    # Load the sql examples for few-shot prompting examples
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

    local_embeddings = HuggingFaceEmbeddings(model_name=HUGGING_FACE_EMBEDDINGS_MODEL)

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
        use_query_checker=False,  # must be False for OpenAI model
        verbose=True,
        return_intermediate_steps=True,
    )


def clear_text():
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""


def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]


if __name__ == "__main__":
    main()
