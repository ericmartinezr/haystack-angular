from haystack.dataclasses import ChatMessage
from agents.angular import angular
from dotenv import load_dotenv

load_dotenv()


def run_agent(query: str):
    print(f"Agent running with query: {query}")
    response = angular.run(data={
        "agent": {
            "messages": [ChatMessage.from_user(query)]
        }
    })

    last_message = response["agent"]["messages"][-1]
    print("\nAgent Response:\n")
    print(last_message.text)


if __name__ == "__main__":
    user_query = """
Generate a simple Angular component that displays a list of items and allows the 
user to add new items to the list.
    """
    run_agent(user_query)
