from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun

ollama_gemma = Ollama(model="gemma")
search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role = 'Senior Research Analyst', 
    goal = 'Uncover cutting-edge developments in AI, machine learning, and data science', 
    backstory = """You are a Senior Research Analyst at a leading tech think tank.
  Your expertise lies in identifying emerging trends and technologies in AI and
  data science. You have a knack for dissecting complex data and presenting
  actionable insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_gemma
)

writer = Agent(role = "writer", 
                   goal="write compelling narratives about new technologies", 
                   backstory="""You are a renowned Writer andContent Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the AI industry, you transform complex concepts into compelling narratives.""",
                   verbose=True,
                   allow_delegation=False,
                   llm=ollama_gemma)

research = Task(description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. Your final answer MUST be a full analysis report.""", agent=researcher)

write = Task(description="""Using the insights from the researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future.""", agent=writer)

crew = Crew(
            agents=[researcher, writer], 
            tasks=[research, write], 
            verbose=2)

results = crew.kickoff()
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print(results)