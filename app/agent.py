from langchain_core.messages import HumanMessage

from app.graph import agent

# This file is now the main entry point for your application.
# It's kept clean and simple.


def invoke_agent(query: str) -> str:
    """A simple function to invoke the agent with a user query."""
    initial_state = {"messages": [HumanMessage(content=query)]}

    final_state = agent.invoke(initial_state)  # type: ignore[arg-type]
    return final_state["messages"][-1].content


if __name__ == "__main__":
    # This allows you to run the agent directly for testing
    print("Welcome to the MLB AI Agent! Type 'exit' to quit.")
    while True:
        user_query = input("> ")
        if user_query.lower() == "exit":
            break

        response = invoke_agent(user_query)
        print(f"\nAgent: {response}\n")
