# Spring-AI-Booking

This project is an exploration of the [Spring AI](https://docs.spring.io/spring-ai/reference/index.html) project.
It accompanies a blog post on [johnsosoka.com](https://www.johnsosoka.com/blog/2024/03/24/Spring-AI.html).

## About

This simple project creates a Hotel Booking Agent which is capable of checking availability, booking rooms, and looking
up reservations. The project is built using the Spring/AI Framework. Multiple functions are imlemented and exposed to 
the Agent that grants it these capabilities.


## Getting Started

### Pre-requisites
* This project requires an OpenAI API key. You can get one [here](https://platform.openai.com/signup).

### Running the project
* Clone the repository
* Set an environment variable `OPENAI_API_KEY` with your OpenAI API key.
* Run the project using `./mvnw spring-boot:run`

### Interacting with the Agent
* A postman collection is provided in the `postman` directory. You can import this collection into Postman and interact with the Agent.
* Alternatively you can use `curl` to interact with the Agent. The following examples show how to interact with the Agent using `curl`:
```shell
`curl -X POST -H "Content-Type: text/plain" -d "Hi, my name is John--Can you see if any rooms are available on February 28, 2025?" http://localhost:8080/ai/booking`
```

The Booking Service has two hardcoded dates:
* January 15, 2025 (available)
* February 28, 2025 (unavailable)

You can converse with the agent to check availability, book a room, and look up reservations via the `/ai/booking` endpoint.

It is important to note that this is an exploratory project. All state is managed in-memory and it can only support one conversation at a time.