from workflow.graph import graph
from workflow.state import State

import logging

logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Explicitly add a stream handler for console output
)

# Silence httpx
logging.getLogger("httpx").setLevel(logging.ERROR)



#
# ADVANCED ROUTING REQUEST
#
logging.info("The first query should route to the advanced model.")

advanced_message = (
    "Please ensure that product ID 18235 has inventory and doesn't have any "
    "availability overrides set. If there are no overrides set, and inventory is zero, "
    "please order a new batch."
)

advanced_state: State = {
    "user_query": advanced_message
}

graph.invoke(advanced_state)



#
# SIMPLE ROUTING REQUEST
#


logging.info("The second query should route to the simple model.")
simple_message = "does product ID 1234 come in red?"
simple_state: State = {
    "user_query": simple_message
}

graph.invoke(simple_state)
