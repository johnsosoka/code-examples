package com.johnsosoka;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.johnsosoka.agent.BlogEditor;
import com.johnsosoka.model.FeedbackItem;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import lombok.SneakyThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;

public class Main {


    public static void main(String[] args) {
        String API_KEY = System.getenv("API_KEY");

        if (API_KEY == null || API_KEY.isEmpty()) {
            System.out.println("API_KEY not set. Exiting...");
            return;
        }

        BlogEditor requirementAssistant = AiServices.builder(BlogEditor.class)
                .chatLanguageModel(OpenAiChatModel.builder()
                        .apiKey(API_KEY)
                        .modelName("gpt-4")
                        .timeout(Duration.ofSeconds(380))
                        .maxRetries(1)
                        .build())
                //.chatMemory(MessageWindowChatMemory.withMaxMessages(5))
                .build();

        String blogPost = readMarkdownFile("/Users/john/code/johnsosoka-com/jscom-blog/website/_posts/blog/2023-10-31-llm-app-patterns.md");

        List<FeedbackItem> technicalFeedback = getTechnicalFeedback(blogPost, requirementAssistant);
        List<FeedbackItem> contentFeedback = getContentEditorFeedback(blogPost, requirementAssistant);

        printFeedback(technicalFeedback);
        printFeedback(contentFeedback);

        System.out.println("done.");
    }

    public static List<FeedbackItem> getTechnicalFeedback(String blogPost, BlogEditor assistant) {
        String technicalFeedback = assistant.provideTechnicalFeedback(blogPost);
        List<FeedbackItem> technicalFeedbackItems = null;
        try {
            technicalFeedbackItems = new ObjectMapper().readValue(technicalFeedback, new TypeReference<List<FeedbackItem>>(){});
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        System.out.println("Technical Feedback Items: " + technicalFeedbackItems.size());
        return technicalFeedbackItems;
    }

    public static List<FeedbackItem> getContentEditorFeedback(String blogPost, BlogEditor assistant) {
        String contentEditorFeedback = assistant.provideContentEditorFeedback(blogPost);
        List<FeedbackItem> contentEditorFeedbackItems = null;
        try {
            contentEditorFeedbackItems = new ObjectMapper().readValue(contentEditorFeedback, new TypeReference<List<FeedbackItem>>(){});
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        System.out.println("Content Editor Feedback Items: " + contentEditorFeedbackItems.size());
        return contentEditorFeedbackItems;
    }

    public static void printFeedback(List<FeedbackItem> feedbackItems) {
        // iterate through both feedback items printing fields
        for (FeedbackItem feedbackItem : feedbackItems) {
            System.out.println("Feedback Type: " + feedbackItem.getFeedbackType());
            System.out.println("Feedback: " + feedbackItem.getFeedback());
            System.out.println("Why the LLM cares: " + feedbackItem.getExplanation());
            System.out.println(feedbackItem.getFeedback());
        }
    }

    @SneakyThrows
    public static String readMarkdownFile(String filePath)   {
        Path path = Paths.get(filePath);
        List<String> lines = Files.readAllLines(path);
        return String.join("\n", lines);
    }
}
