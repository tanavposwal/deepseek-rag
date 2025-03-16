from crewai import Agent, Task, Crew, Process, LLM


def init_crew(tool):
    llm = LLM(model="ollama/gemma3", base_url="http://localhost:11434")
    researcher = Agent(
        role="Research Agent",
        goal="Search through the PDF to find relevant answers",
        allow_delegation=False,
        verbose=True,
        backstory=(
            """
            The research agent is adept at searching and 
            extracting data from documents, ensuring accurate and prompt responses.
            """
        ),
        tools=[tool],
        llm=llm,
    )

    writer = Agent(
        role="Professional Writer",
        goal="Write professional emails based on the research agent's findings",
        allow_delegation=False,
        verbose=True,
        backstory=(
            """
            The professional writer agent has excellent writing skills and is able to craft 
            clear and concise emails based on the provided information.
            """
        ),
        llm=llm,
    )

    task = Task(
        description=(
            """
            Answer the customer's questions based on the PDF.

            Here is the customer's question:
            {input}
            """
        ),
        expected_output="""
            Provide clear and accurate answers to the customer's questions based on 
            the content of the PDF.
            """,
        tools=[tool],
        agent=researcher,
    )

    crew = Crew(
        tasks=[task],
        agents=[researcher, writer],
        process=Process.sequential,
    )

    return crew
