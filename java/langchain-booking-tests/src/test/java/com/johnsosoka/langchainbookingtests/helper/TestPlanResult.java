package com.johnsosoka.langchainbookingtests.helper;

import lombok.*;


@Builder
@Data
public class TestPlanResult {

    private Boolean allTestsPassed;
    private String testPlan;
    private String testPlanResults;

}
