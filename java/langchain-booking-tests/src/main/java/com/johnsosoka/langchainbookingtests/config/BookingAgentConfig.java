package com.johnsosoka.langchainbookingtests.config;

import com.johnsosoka.langchainbookingtests.agent.BookingAgent;
import com.johnsosoka.langchainbookingtests.tool.BookingTools;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class BookingAgentConfig {

    @Value("${openai.api-key}")
    String apiKey;

    @Bean
    public ChatLanguageModel chatLanguageModel() {
        return OpenAiChatModel.builder()
                .apiKey(apiKey)
                .build();
    }

    @Bean
    public BookingAgent bookingAgent(BookingTools bookingTools, ChatLanguageModel chatLanguageModel) {
        return AiServices.builder(BookingAgent.class)
                .chatLanguageModel(chatLanguageModel)
                .tools(bookingTools)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(50))
                .build();
    }

}
