import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title="🦜🔗 Ask the CSV App")
st.title("🦜🔗 Ask the CSV App")

st.write(
    """
This app allows you to upload a CSV file and ask questions about the data using an OpenAI language model. Follow these steps to get started:

1. **Upload a CSV file**: Use the file uploader below to upload a CSV file containing the data you want to query.
2. **Select or enter a query**: Once you have uploaded a file, you can select an example query from the dropdown list or choose 'Other' to enter your own custom query.
3. **Enter your OpenAI API key**: You will need to enter your OpenAI API key to use the language model. If you don't have an API key, you can sign up for one on the OpenAI website.
4. **View the results**: After entering your query and API key, the app will generate a response to your query using the OpenAI language model and display it below.
"""
)


def load_csv(input_csv):
    """
    Load a CSV file into a Pandas DataFrame and display it in an expandable section of the app.

    :param input_csv: The input CSV file to be loaded.
    :return: A Pandas DataFrame containing the data from the input CSV file.
    """
    try:
        df = pd.read_csv(input_csv)
        with st.expander("See DataFrame"):
            st.write(df)
        return df
    except Exception as e:
        st.error(
            f"Error loading CSV file: {e}. Please make sure you have uploaded a valid CSV file."
        )


def generate_response(csv_file, input_query):
    """
    Generate a response to a user-provided query using an OpenAI language model and a Pandas DataFrame Agent.

    :param csv_file: The uploaded CSV file containing the data to be queried.
    :param input_query: The user-provided query to be answered.
    :return: A Streamlit success element containing the generated response.
    """
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0613",
            temperature=0.2,
            openai_api_key=openai_api_key,
        )
        df = load_csv(csv_file)
        # Create Pandas DataFrame Agent
        agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
        )
        # Perform Query using the Agent
        response = agent.run(input_query)
        return st.success(response)
    except Exception as e:
        st.error(
            f"Error generating response: {e}. Please check your OpenAI API key and make sure it is valid."
        )


# Input widgets
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="Upload a CSV file containing the data you want to query.",
)
question_list = [
    "How many rows and columns are there in the data?",
    "What are the names and data types of the columns?",
    "Are there any missing or null values in the data?",
    "What is the range of values for each column?",
    "Are there any duplicate rows in the data?",
    "Other",
]
query_text = st.selectbox(
    "Select or enter a query:",
    question_list,
    disabled=not uploaded_file,
    help="Select an example query or choose 'Other' to enter your own custom query.",
)
openai_api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    disabled=not (uploaded_file and query_text),
    help="Enter your OpenAI API key. If you don't have one, you can sign up for one on the OpenAI website.",
)

# App logic
if query_text is "Other":
    query_text = st.text_input(
        "Enter your custom query:",
        placeholder="Enter query here ...",
        disabled=not uploaded_file,
        help="Enter your own custom query here.",
    )
if not openai_api_key.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
if openai_api_key.startswith("sk-") and (uploaded_file is not None):
    st.header("Results")
    generate_response(uploaded_file, query_text)
