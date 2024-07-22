package com.johnsosoka.langchainbookingtests.helper;


import lombok.Builder;
import lombok.extern.slf4j.Slf4j;

@Builder
@Slf4j
public class MultiPassEvaluator {

    private TestEvaluationAgent testEvaluationAgent;
    // The total number of times to evaluate the result
    private Integer passCount;

    public Boolean evaluate(String condition, String result) {
        Boolean evaluationResult = false;
        int successCount = 0;
        for (int i = 0; i < passCount; i++) {
            boolean evaluation = testEvaluationAgent.evaluate(condition, result);
            if (evaluation) {
                successCount++;
                log.info("Evaluation {} passed", i);
            } else {
                log.info("Evaluation {} failed", i);
            }
        }
        // If more than half of the evaluations are successful, then the test is considered successful
        return successCount >= passCount / 2;
    }

}
