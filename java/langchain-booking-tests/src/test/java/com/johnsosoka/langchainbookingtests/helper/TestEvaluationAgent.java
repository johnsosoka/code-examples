package com.johnsosoka.langchainbookingtests.helper;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

import java.util.List;

public interface TestEvaluationAgent {

    @SystemMessage({
            "You purpose is to evaluate the results of a test. You will be employed in a unit testing environment, ",
            "and must critically evaluate the provided conditions and results to determine if the test has passed or ",
            "failed. Consider a passing test True, and a failing test False."
    })
    @UserMessage({
            "Evaluate the following:\n",
            "Condition: {{condition}}\n",
            "-----\n",
            "Results: {{result}}",
    })
    public Boolean evaluate(@V("condition") String condition, @V("result") String result);

    @SystemMessage({
            "You are a world class QA engineer, your job is to test the system and ensure that it is working as expected.",
            "You will be provided with an explanation of the System's behavior and you must carefully write test cases to",
            "ensure that the system meets the expected behavior."
    })
    @UserMessage({
            "Write test cases for the following system behavior:\n",
            "System Behavior: {{systemBehavior}}\n"
    })
    public List<TestCase> writeTestCases(@V("systemBehavior") String systemBehavior);
}
