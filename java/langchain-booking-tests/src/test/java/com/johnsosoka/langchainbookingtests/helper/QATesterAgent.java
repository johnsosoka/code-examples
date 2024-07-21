package com.johnsosoka.langchainbookingtests.helper;

import dev.langchain4j.service.SystemMessage;

public interface QATesterAgent {

    @SystemMessage({
            "You are a world class QA engineer, your job is to test the system and ensure that it is working as expected.",
            "You will be provided with a set of conditions and you must determine if the system meets those conditions.",
            "You will be provided with a set of test cases and a tool exposing you to the system being tested."
    })
    public Boolean test(String conditions);


}
