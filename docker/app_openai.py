# Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL and OpenAI's LLM models via their API.
# Author: Gary A. Stafford (garystaf@amazon.com)
# Date: 2023-07-12
# Application expects the following environment variables (adjust for your environment):
# export OPENAI_API_KEY="sk-<your_api_key>""
# export REGION_NAME="us-east-1"
# export MODEL_NAME="gpt-3.5-turbo"
# Usage: streamlit run app_openai.py --server.runOnSave true

import ast
import json
import logging
import os

import altair as alt
import boto3
import pandas as pd
import streamlit as st
import yaml
from botocore.exceptions import ClientError
from langchain import (FewShotPromptTemplate, PromptTemplate, SQLDatabase,
                       SQLDatabaseChain)
from langchain.chains.sql_database.prompt import (PROMPT_SUFFIX,
                                                  _postgres_prompt)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import \
    SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from streamlit_chat import message

REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")


def main():
    st.set_page_config(
        page_title="Natural Language Query (NLQ) Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    NO_ANSWER_MSG = "I'm sorry, I was not able to answer your question."

    os.environ["OPENAI_API_KEY"] = set_openai_api_key(REGION_NAME)

    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.0,
        verbose=True,
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

    # define stearmlit tabs
    tab1, tab2 = st.tabs(["Chat", "Details"])

    with tab1:
        # define streamlit columns
        col1, col2 = st.columns([6, 1], gap="medium")

        with col1:
            with st.container():
                st.markdown("## Natural Language Query (NLQ) Demonstration")
                st.markdown(
                    "##### Ask questions about The Museum of Modern Art (MoMA) Collection using natural language."
                )
                with st.expander("Click here for sample questions..."):
                    st.markdown(
                        """
                        * How many artists are there in the collection?
                        * How many pieces of artwork are there in the collection?
                        * How many artists are there whose nationality is Italian?
                        * How many artworks are by the artist Claude Monet?
                        * How many artworks are classified as paintings?
                        * How many artworks were created by Spanish artists?
                        * How many artist names start with the letter 'M'?
                        ---
                        * How many artists are deceased as a percentage of all artists?
                        * Who is the most prolific artist in the collection? What is their nationality?
                        * What nationality of artists created the most artworks in the collection?
                        * What is the ratio of male to female artists? Return as a ratio.
                        * How many artworks were produced during the First World War, which are classified as paintings?
                        * What are the five oldest artworks in the collection? Return the title and date for each.
                        * Return the artwork for Frida Kahlo in a numbered list, including the title and date.
                        * What is the count of artworks by classification? Return the first ten in descending order. Don't include Not_Assigned.
                        * What are the ten artworks by European artist, with a data? Write Python code to output them with Matplotlib as a table. Include a header row and font size of 12.
                        ---
                        * Give me a recipe for chocolate cake.
                        * Don't write a SQL query. Don't use the database. Tell me who won the 2022 FIFA World Cup final?
                    """
                    )

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
                            if (i >= 0) and (
                                st.session_state["generated"][i] != NO_ANSWER_MSG
                            ):
                                with st.chat_message(
                                    "assistant", avatar="app/static/bot-64px.png"
                                ):
                                    st.write(st.session_state["generated"][i]["result"])
                                with st.chat_message("user", avatar="app/static/man-64px.png"):
                                    st.write(st.session_state["past"][i])
                            else:
                                with st.chat_message(
                                    "assistant", avatar="app/static/bot-64px.png"
                                ):
                                    st.write(NO_ANSWER_MSG)
                                with st.chat_message("user", avatar="app/static/man-64px.png"):
                                    st.write(st.session_state["past"][i])
        with col2:
            with st.container():
                st.button("clear chat", on_click=clear_session)

    with tab2:
        with st.container():
            st.markdown("# Under the Hood")

            position = len(st.session_state["generated"]) - 1
            if (position >= 0) and (
                st.session_state["generated"][position] != NO_ANSWER_MSG
            ):
                st.markdown("OpenAI Model:")
                st.code(MODEL_NAME, language="text")

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
                df = None
                if len(data[0]) == 2:
                    st.markdown("Table:")
                    df = pd.DataFrame(data, columns=["Category", "Metric"])
                    df = df.astype({"Metric": "str"})
                    df.sort_values(by=["Metric"])
                    df

                if len(data[0]) == 2:
                    # ax = df.plot.bar()
                    # st.bar_chart(data=df, x='Category', y='Count', use_container_width=True)

                    st.markdown("Chart:")
                    df = df.astype({"Metric": "Int64"})
                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Category", sort=None),
                            y="Metric",
                        )
                        .interactive()
                    )
                    st.altair_chart(chart, theme="streamlit", use_container_width=True)
            else:
                st.markdown("Nothing to see here...")
        with st.container():
            st.markdown("""---""")
            st.markdown(
                "![](app/static/github-24px-wht.png) [Submit feature request or bug report](https://github.com/aws-solutions-library-samples/guidance-for-natural-language-queries-of-relational-databases-on-aws/issues)"
            )
            st.markdown(
                "![](app/static/github-24px-wht.png) [Source code](https://github.com/aws-solutions-library-samples/guidance-for-natural-language-queries-of-relational-databases-on-aws)"
            )


def set_openai_api_key(region_name):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    openai_api_key = None

    try:
        secret = client.get_secret_value(SecretId="/nlq/OpenAIAPIKey")
        openai_api_key = secret["SecretString"]
    except ClientError as e:
        logging.error(e)
        raise e

    return openai_api_key


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
