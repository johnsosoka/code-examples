package com.johnsosoka.langchainbookingtests.helper;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface QATesterAgent {

    @SystemMessage({
            "You are a world class QA engineer, your job is to test the system and ensure that it is working as expected.",
            "You will be provided with a test plan, and it is your job to execute each test case individually and determine",
            "if the system is working as expected.",
            "You will act as a customer interacting with a chatbot system to test the system's behavior.",
    })
    public String test(String testCases);

    @SystemMessage({
            "You are a world class QA engineer, your job is to test the system and ensure that it is working as expected.",
            "You will be provided with an explanation of the System's behavior and you must carefully write test cases to",
            "ensure that the system meets the expected behavior. Your test cases should be a detailed description for usage",
            "by a different language model.",
            "The System being tested is another Large Language Model, so the inputs and expected outputs can be in natural language.",
            "Account for this possible variability in the rigidity of evaluation criteria."
    })
    @UserMessage({
            "Write test cases for the following system behavior:\n",
            "System Behavior: {{systemBehavior}}\n"
    })
    public String writeTestCases(@V("systemBehavior") String systemBehavior);

    @SystemMessage({
            "You must carefully evaluate the results of the test plan to determine if the system is working as expected.",
            "In the event of any failures, the result should be false. Otherwise, the result should be true."
    })
    @UserMessage("Evaluate the following test execution results: {{it}}")
    public Boolean evaluateResults(String testResults);

}
