package com.johnsosoka.springaibooking.service;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ConversationServiceTest {

    private ConversationService conversationService;

    @BeforeEach
    void setUp() {
        conversationService = new ConversationService();
    }

    @Test
    void testAddMessage() {
        // Create a user message
        Message userMessage = new UserMessage("Hello, how are you?");

        // Add the user message to the conversation
        List<Message> updatedMessages = conversationService.addMessage(userMessage);

        // Assert that the updated messages list contains the user message
        assertEquals(1, updatedMessages.size());
        assertTrue(updatedMessages.contains(userMessage));
    }

    @Test
    void testAddMultipleMessages() {
        // Create user messages
        Message userMessage1 = new UserMessage("Hello!");
        Message userMessage2 = new UserMessage("How can I book a room?");

        // Create assistant messages
        Message assistantMessage1 = new AssistantMessage("Hi there! How can I assist you today?");
        Message assistantMessage2 = new AssistantMessage("To book a room, please provide your desired check-in and check-out dates.");

        // Add the messages to the conversation
        conversationService.addMessage(userMessage1);
        conversationService.addMessage(assistantMessage1);
        conversationService.addMessage(userMessage2);
        List<Message> updatedMessages = conversationService.addMessage(assistantMessage2);

        // Assert that the updated messages list contains all the added messages in the correct order
        assertEquals(4, updatedMessages.size());
        assertEquals(userMessage1, updatedMessages.get(0));
        assertEquals(assistantMessage1, updatedMessages.get(1));
        assertEquals(userMessage2, updatedMessages.get(2));
        assertEquals(assistantMessage2, updatedMessages.get(3));
    }

    @Test
    void testGetAllMessages() {
        // Create user messages
        Message userMessage1 = new UserMessage("Hello!");
        Message userMessage2 = new UserMessage("How can I book a room?");

        // Create assistant messages
        Message assistantMessage1 = new AssistantMessage("Hi there! How can I assist you today?");
        Message assistantMessage2 = new AssistantMessage("To book a room, please provide your desired check-in and check-out dates.");

        // Add the messages to the conversation
        conversationService.addMessage(userMessage1);
        conversationService.addMessage(assistantMessage1);
        conversationService.addMessage(userMessage2);
        conversationService.addMessage(assistantMessage2);

        // Retrieve all messages from the conversation
        List<Message> allMessages = conversationService.getAllMessages();

        // Assert that all messages are retrieved in the correct order
        assertEquals(4, allMessages.size());
        assertEquals(userMessage1, allMessages.get(0));
        assertEquals(assistantMessage1, allMessages.get(1));
        assertEquals(userMessage2, allMessages.get(2));
        assertEquals(assistantMessage2, allMessages.get(3));
    }

    @Test
    void testGetAllMessagesEmptyConversation() {
        // Retrieve all messages from an empty conversation
        List<Message> allMessages = conversationService.getAllMessages();

        // Assert that an empty list is returned
        assertTrue(allMessages.isEmpty());
    }
}