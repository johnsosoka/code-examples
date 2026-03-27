package com.johnsosoka.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class FeedbackItem {

    @JsonProperty("feedback_type")
    private String feedbackType;

    @JsonProperty("original_quote")
    private String originalQuote;

    @JsonProperty("feedback")
    private String feedback;

    @JsonProperty("explanation")
    private String explanation;

    @JsonProperty("follow_up")
    private FollowUp followUp;

    @Data
    public static class FollowUp {

        @JsonProperty("action_required")
        private String actionRequired;

        @JsonProperty("additional_info")
        private String additionalInfo;
    }
}

