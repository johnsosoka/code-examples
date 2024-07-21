package com.johnsosoka.langchainbookingtests.service;

import com.johnsosoka.langchainbookingtests.helper.MultiPassEvaluator;
import com.johnsosoka.langchainbookingtests.helper.TestEvaluationAgent;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.service.AiServices;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@Slf4j
class ChatServiceTestIT {

    @Autowired
    private ChatService chatService;

    @Autowired
    private ChatLanguageModel chatLanguageModel;

    private TestEvaluationAgent testEvaluationAgent;

    private MultiPassEvaluator multiPassEvaluator;

    @BeforeEach
    public void setUp(){
        testEvaluationAgent = provisionEvaluationAgent();
        multiPassEvaluator = MultiPassEvaluator.builder()
                .testEvaluationAgent(testEvaluationAgent)
                .passCount(3)
                .build();
    }

    @Test
    public void checkAvailability_withTestEvaluationAgent() {
        String response = chatService.chat("Is the hotel available on 2025-02-28?");
        log.info("Response: {}", response);

        String condition = "It should be determined that there are no hotel rooms available on 2025-02-28";
        Boolean evaluationResult = testEvaluationAgent.evaluate(condition, response);
        assertTrue(evaluationResult);
    }

    @Test
    public void checkAvailability_withMultiPassEvaluator() {
        String response = chatService.chat("Is the hotel available on 2025-02-28?");
        log.info("Response: {}", response);

        String condition = "It should be determined that there are no hotel rooms available on 2025-02-28";
        Boolean evaluationResult = multiPassEvaluator.evaluate(condition, response);
        assertTrue(evaluationResult);
    }


    private TestEvaluationAgent provisionEvaluationAgent() {
        return AiServices.builder(TestEvaluationAgent.class)
                .chatLanguageModel(chatLanguageModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    @Test
    public void checkAvailability() {
        String response = chatService.chat("Is the hotel available on 2025-02-28");
        log.info("Response: {}", response);
        assertTrue(response.contains("not available"));
    }
}