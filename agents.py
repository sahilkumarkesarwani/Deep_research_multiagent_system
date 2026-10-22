from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrap_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatMistralAI(model="ministral-3b-2512", temperature=0)

def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search]
    )

def build_scrap_agent():
    return create_agent(
        model = llm,
        tools = [scrap_url]
    )

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

        Topic: {topic}
        Research Gathered: {research}

        Structure the report as:
        - Introduction
        - Key Findings (minimum 3 well-explained points)
        - Conclusion
        - Sources (list all URLs found in the research)

        Be detailed, factual and professional."""),
    ])

writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([

    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

        Report: {report}
        Respond in this exact format:

        Score: X/10

        Strengths:
        - ...
        - ...

        Areas to Improve:
        - ...
        - ...

        One line verdict:
        ..."""),
    ])


critic_chain = critic_prompt | llm | StrOutputParser()



qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You answer follow-up questions about a research report. Base your answer on "
     "the context below. If the context does not cover the question, say so clearly "
     "instead of guessing. Be concise and use markdown.\n\nContext:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

qa_chain = qa_prompt | llm | StrOutputParser()