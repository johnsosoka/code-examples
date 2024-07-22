# LangChain4J-Agentic-Unit-Tests

This project demonstrates a handful of strategies for unit testing an LLM-Driven application with non-deterministic output.
It accompanies a blog post on [johnsosoka.com](https://www.johnsosoka.com/blog/2024/07/21/unit-testing-llms.html).

## About

This project contains a simple Hotel Booking Agent which is capable of checking availability, booking rooms, and looking
up reservations. The primary focus of the project is to create unit tests to evaluate the behavior of the Hotel Booking
Agent. This project was built using LangChain4J.

## Getting Started

### Pre-requisites
* This project requires an OpenAI API key. You can get one [here](https://platform.openai.com/signup).

### Running the project
* Clone the repository
* Set an environment variable `OPENAI_API_KEY` with your OpenAI API key.
* Run the project using `./mvnw spring-boot:run`

### Running the tests
* Run the tests using `./mvnw test`
