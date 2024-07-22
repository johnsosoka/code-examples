package com.johnsosoka.langchainbookingtests.service;

import com.johnsosoka.langchainbookingtests.helper.*;
import com.johnsosoka.langchainbookingtests.helper.tool.BookingAgentTool;
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

    @Autowired
    private BookingAgentTool bookingAgentTool;

    private TestEvaluationAgent testEvaluationAgent;

    private MultiPassEvaluator multiPassEvaluator;

    private QATesterAgent qaTesterAgent;

    private AgenticQA agenticQA;


    @BeforeEach
    public void setUp(){
        testEvaluationAgent = provisionEvaluationAgent();
        multiPassEvaluator = MultiPassEvaluator.builder()
                .testEvaluationAgent(testEvaluationAgent)
                .passCount(3)
                .build();
        agenticQA = new AgenticQA(provisionQATesterAgent());
    }

    @Test
    public void testPlanCreationTest() {
        String systemDescription = """
                The system is a simple hotel booking agent. The agent should have the ability to:
                - Check the availability of a hotel room for a given date
                - Book a hotel room for a guest (check in & check out date required)
                - Lookup a booking by guest name
                
                The system has the following preconditions:
                - The system has a hotel with 1 room available on 2025-01-15
                - The system has a hotel with 0 rooms available on 2025-02-28
                - All other dates should be considered unavailable
                """;

        TestPlanResult testPlanResult = agenticQA.generateAndExecuteTestPlan(systemDescription);
        log.info("Test Plan: \n{}", testPlanResult.getTestPlan());
        log.info("Test Plan Results: \n{}", testPlanResult.getTestPlanResults());
        assertTrue(testPlanResult.getAllTestsPassed());
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

    private QATesterAgent provisionQATesterAgent() {
        return AiServices.builder(QATesterAgent.class)
                .chatLanguageModel(chatLanguageModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(50))
                .tools(bookingAgentTool)
                .build();
    }


    @Test
    public void checkAvailability() {
        String response = chatService.chat("Is the hotel available on 2025-02-28");
        log.info("Response: {}", response);
        assertTrue(response.contains("not available"));
    }
}