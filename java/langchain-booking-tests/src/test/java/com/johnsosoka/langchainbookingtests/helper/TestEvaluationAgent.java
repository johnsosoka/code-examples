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

}
