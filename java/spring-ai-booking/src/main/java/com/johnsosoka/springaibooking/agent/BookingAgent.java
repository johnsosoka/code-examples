package com.johnsosoka.springaibooking.agent;

import com.johnsosoka.springaibooking.service.ConversationService;
import com.johnsosoka.springaibooking.tool.BookRoomTool;
import com.johnsosoka.springaibooking.tool.CheckAvailabilityTool;
import com.johnsosoka.springaibooking.tool.FindBookingTool;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.function.FunctionCallbackWrapper;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
public class BookingAgent {

    private final ChatClient chatClient;
    private final CheckAvailabilityTool checkAvailabilityTool;
    private final FindBookingTool findBookingTool;
    private final BookRoomTool bookRoomTool;
    private final ConversationService conversationService;

    /**
     * When the BookingAgent is created, we will define the agent's role as a SystemMessage at the top of the conversation.
     */
    @PostConstruct
    public void defineAgentProfile() {
        String agentProfile = "You are a booking agent for an online hotel. You are here to help customers book rooms and check availability.";
        SystemMessage systemMessage = new SystemMessage(agentProfile);
        conversationService.addMessage(systemMessage);
    }

    public String handleMessage(String message) {
        // Add the user message to the conversation
        UserMessage latestMessage = new UserMessage(message);
        conversationService.addMessage(latestMessage);

        List<Message> messages = conversationService.getAllMessages();
        var promptOptions = getPromptOptions();

        ChatResponse response = chatClient.call(new Prompt(messages, promptOptions));
        // Add the assistant response to the conversation
        conversationService.addMessage(response.getResult().getOutput());

        // Return the assistant response
        return response.getResult().getOutput().getContent();

    }

    /**
     * Expose function callbacks to the OpenAI chat client
     * @return
     */
    private OpenAiChatOptions getPromptOptions() {
        return OpenAiChatOptions.builder()
                .withFunctionCallbacks(List.of(FunctionCallbackWrapper.builder(checkAvailabilityTool)
                        .withName("CheckAvailability")
                        .withDescription("Check the availability of rooms for a specific date")
                        .withResponseConverter((response) -> "" + response.available())
                        .build()))
                .withFunctionCallbacks(List.of(FunctionCallbackWrapper.builder(bookRoomTool)
                        .withName("BookRoom")
                        .withDescription("Helpful for booking a room for a guest for a specific check-in and check-out date")
                        .withResponseConverter((response) -> response.bookingStatus())
                        .build()))
//                .withFunctionCallbacks(List.of(FunctionCallbackWrapper.builder(findBookingTool)
//                        .withName("FindBooking")
//                        .withDescription("Helpful to determine if a guest has booked a room")
//                        .withResponseConverter((response) -> response.booking())
//                        .build()))
                .build();
    }
}
