package com.johnsosoka.springaibooking.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Service;

import org.springframework.ai.chat.messages.Message;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Service class to manage a single conversation as an ordered list of messages.
 * This class is designed to be used as a singleton bean in a Spring application.
 * It provides methods to add messages to the conversation and retrieve all messages.
 * The conversation data is stored in a thread-safe manner to ensure data consistency across requests.
 */
@Service
@Scope("singleton")
@Slf4j
public class ConversationService {
    private List<Message> messageList = Collections.synchronizedList(new ArrayList<>());

    /**
     * Adds a message to the conversation.
     *
     * @param message the message to be added to the conversation
     * @return the updated list of messages in the conversation
     */
    public synchronized List<Message> addMessage(Message message) {
        messageList.add(message);
        log.info("Added message to conversation: {}, total messages: {}", message, messageList.size());
        return new ArrayList<>(messageList);
    }

    /**
     * Retrieves all messages in the conversation.
     *
     * @return the list of messages in the conversation
     */
    public synchronized List<Message> getAllMessages() {
        log.info("Retrieved all {} messages", messageList.size());
        return new ArrayList<>(messageList);
    }
}