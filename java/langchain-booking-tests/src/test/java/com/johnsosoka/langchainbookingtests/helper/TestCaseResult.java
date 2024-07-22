package com.johnsosoka.langchainbookingtests.helper;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Data
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class TestCaseResult {

    private String testCase;
    private Boolean result;
    private String reasoning;

}
