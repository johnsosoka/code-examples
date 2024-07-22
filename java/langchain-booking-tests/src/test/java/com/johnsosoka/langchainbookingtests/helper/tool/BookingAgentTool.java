package com.johnsosoka.langchainbookingtests.helper.tool;

import com.johnsosoka.langchainbookingtests.service.ChatService;
import dev.langchain4j.agent.tool.Tool;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * This Tool class wraps the `ChatService` class, so that the target BookingAgent is exposed as a tool
 * to our QA Tester Agent.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class BookingAgentTool {

    private final ChatService chatService;

    @Tool("Interact with the Booking Agent -- Useful for testing the Booking Agent system")
    public String interactWithBookingAgent(String message) {
//        log.info("QA Agent Message: {}", message);
        System.out.println("QA Agent Message - " + message);
        String response = chatService.chat(message);
//        log.info("Booking Agent Response: {}", response);
        System.out.println("Booking Agent Response - " + response);
        return response;
    }

}
