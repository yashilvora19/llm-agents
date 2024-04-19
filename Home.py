import streamlit as st
import sqlite3
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# RUN THE FILE
# python -m streamlit run Home.py OR
# streamlit run Home.py

st.set_page_config(page_title="Analyzing Wildfire Data")

st.title("Using an LLM agent to analyze wildfire data")

st.markdown(
    """
    This project assigned to me by Aira Technologies aims to build an LLM agent to chat with a wildfire dataset. I am using the OpenAI API 
    and GPT-4 model to communicate with my data.

    Since my data was stored as a SQLite database, I first wrote some queries to get an idea of what it looked like.

    Following this, I converted it to a Pandas dataframe and made use of the `create_pandas_dataframe_agent` in the langchain 
    framework in order to build my LLM agent. After getting my API key and loading some money into my account, I ran some prompts 
    through this agent and got some interesting results. Scroll to the end of the page to see them!

    """
)



conn = sqlite3.connect('archive/wildfires.sqlite')
df = pd.read_sql_query("SELECT * FROM fires", conn)

key = st.text_input("Paste your OpenAI API key here: ")

# Replace with your OpenAI API key
# openai_api_key = key
openai_api_key = 'sk-proj-Kig0RVOZ0wCBtvohuLhUT3BlbkFJECatHKWAPhCyKmy5BieQ'

with st.expander("🔎Dataframe Preview"):
    st.write(df.head(5))

chat = ChatOpenAI(openai_api_key = openai_api_key, model_name = 'gpt-4', temperature = 0.0)
agent = create_pandas_dataframe_agent(chat, df, verbose=True)

query = st.text_area("Now test this agent here!")

clicked = st.button("Generate response!")

if clicked: 
    out = agent.run(query)
    st.write(out)



st.subheader('Prompting the agent with some interesting questions!')

st.image('imgs/q1.png', caption='Started off with a simple question')

st.image('imgs/q2.png', caption='Trying to plot a bargraph')

st.image('imgs/q3.png', caption='Trying a harder question- the agent seemed to struggle with this one!')

st.image('imgs/q4.png', caption='Asking a slightly harder question- pretty impressive job')
