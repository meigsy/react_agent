import os

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tools_scrape import url_to_text


def main(question: str,
         max_iterations: int,
         max_execution_time: int):
    template = """You are a helpful question answerer. Answer the question to the best of your ability. 
    
    You have access to the following tools:
    
    {tools}
    
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {question}
    Thought: {agent_scratchpad}
    """

    prompt_template = PromptTemplate.from_template(template)

    tools = [url_to_text]
    tools_description = render_text_description(list(tools))  # Make sure this function and usage are correct
    tool_names = ", ".join([tool.name for tool in tools])

    api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(
        open_ai_key=api_key,
        model="gpt-4-0125-preview",
        temperature=0,
        timeout=60)

    llm_chain = LLMChain(llm=llm, prompt=prompt_template.partial(
        tools=tools_description,
        tool_names=tool_names  # Corrected here
    ))

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    try:
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            early_stopping_method="generate",
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
            handle_parsing_errors=True
        )

        agent_executor.invoke(
            {
                "question": question
            }
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    main(question="What is the capital of France?",
         max_iterations=3,
         max_execution_time=60)
