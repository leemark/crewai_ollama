from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

ollama_gemma = Ollama(model="gemma")

researcher = Agent(
    role = "researcher", 
    goal = "research new technologies", 
    backstory = "you are a researcher who is an expert in new technologies",
    verbose=True,
    allow_delegation=False,
    llm=ollama_gemma
)
writer = Agent(role = "writer", 
                   goal="write about new technologies", 
                   backstory="you are a blogger who is an expert in new technologies",
                   verbose=True,
                   allow_delegation=False,
                   llm=ollama_gemma)
research = Task(description="research the latest AI news", agent=researcher)
write = Task(description="write a blog post about the latest AI news and trends", agent=writer)

crew = Crew(
            agents=[researcher, writer], 
            tasks=[research, write], 
            verbose=True)

results = crew.kickoff()

print(results)