package com.johnsosoka.prompt;

import dev.langchain4j.model.input.structured.StructuredPrompt;
import lombok.Builder;



@Builder
@StructuredPrompt({"You are a world-class editor helping a friend edit their personal tech blog.",
        "Your goal is to carefully analyze the blog post and provide feedback to your friend.",
        "Notes from Author: {{authorNotes}}",
        "Blog Post Content: {{blogPost}}",
})
public class DynamicPrompt {

    private String authorNotes;
    private String blogPost;

}
