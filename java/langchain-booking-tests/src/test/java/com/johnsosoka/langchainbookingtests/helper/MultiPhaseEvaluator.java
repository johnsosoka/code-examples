package com.johnsosoka.langchainbookingtests.helper;


import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@RequiredArgsConstructor
@Slf4j
public class MultiPhaseEvaluator {

    private final TestAgent testAgent;

    /**
     * Generate and execute a test plan for a given System Description.
     * If the QATesterAgent supplied is equipped with the appropriate tools, it will generate a test plan,
     * execute the test plan, and evaluate the results. It is intended for usage against an LLM-Driven Agent.
     * @param systemDescription
     * @return
     */
    public TestPlanResult generateAndExecuteTestPlan(String systemDescription) {
        String testCases = testAgent.writeTestCases(systemDescription);
        String testPlanResults = testAgent.test(testCases);
        Boolean testPlanResult = testAgent.evaluateResults(testPlanResults);
        return TestPlanResult.builder()
                .testPlan(testCases)
                .testPlanResults(testPlanResults)
                .allTestsPassed(testPlanResult)
                .build();
    }
}
