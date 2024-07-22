package com.johnsosoka.langchainbookingtests.helper;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TestCase {

    private String scenario;
    private Boolean expectedResult;

}
