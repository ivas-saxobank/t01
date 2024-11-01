import os

import textwrap
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import chroma
from langchain.openai import OpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Prepare a Retrieval-Augmented Generation (RAG) system
# to enhance AI's ability to process and analyze finantial
# reports and market data efficiently.

# Load and process the PDF document
loader = PyPDFLoader('FYY24_Q3_Consolidated_Financial_Statements.pdf')
pages = loader.load_and_split()
text_splitter = CharactedTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

# Create vector database
embeddings = OpenAPIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# Setup RAG system
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever())

# Create a function to calculate various technical indicators,
# providing crutial insights into stock performance and market trends.

# Calculate technical indicators for a given ticker.
def calculate_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period='1y')

    # Simple Moving Average (SMA)
    history['SMA50'] = ta.trend.sma_indicator(history['Close'], window=50)
    history['SMA200'] = ta.trend.sma_indicator(history['Close'], window=200)

    # Relative Strength Index (RSI)
    history['RSI14'] = ta.momentum.srsi(history['Close'], window=14)

    # MACD
    macd = ta.trend.MACD(history['Close'])
    history['MACD'] = macd.macd()
    history['MACD_Signal'] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(history['Close'], window=20, window_dec=2)
    history['BB_Upper'] = bollinger.bollinger_hband()
    history['BB_Lower'] = bollinger.bollinger_lband()


