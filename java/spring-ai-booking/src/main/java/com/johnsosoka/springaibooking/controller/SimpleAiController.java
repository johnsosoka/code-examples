package com.johnsosoka.springaibooking.controller;

import com.johnsosoka.springaibooking.tool.BookRoomTool;
import com.johnsosoka.springaibooking.tool.CheckAvailabilityTool;
import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.function.FunctionCallbackWrapper;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;




@RestController
public class SimpleAiController {

    private final ChatClient chatClient;
    private final CheckAvailabilityTool checkAvailabilityTool;
    private final BookRoomTool bookRoomTool;

    @Autowired
    public SimpleAiController(ChatClient chatClient, CheckAvailabilityTool checkAvailabilityTool, BookRoomTool bookRoomTool) {
        this.chatClient = chatClient;
        this.checkAvailabilityTool = checkAvailabilityTool;
        this.bookRoomTool = bookRoomTool;
    }

    @GetMapping("/ai/simple")
    public Map<String, String> completion(@RequestParam(value = "message", defaultValue = "Can you book a room for John on January 15th 2025?") String message) {

        UserMessage userMessage = new UserMessage(message);

        var promptOptions = OpenAiChatOptions.builder()
                .withFunctionCallbacks(List.of(FunctionCallbackWrapper.builder(checkAvailabilityTool)
                        .withName("CheckAvailability")
                        .withDescription("Check the availability of rooms for a specific date")
                        .withResponseConverter((response) -> "" + response.available())
                        .build()))
                .withFunctionCallbacks(List.of(FunctionCallbackWrapper.builder(bookRoomTool)
                        .withName("BookRoom")
                        .withDescription("Book a room for a guest for a specific check-in and check-out date")
                        .withResponseConverter((response) -> response.bookingStatus())
                        .build()))
                .build();

        ChatResponse response = chatClient.call(new Prompt(List.of(userMessage), promptOptions));
        return Map.of("generation", response.getResult().toString());
    }
}