package com.johnsosoka.langchainbookingtests.agent;

import dev.langchain4j.service.SystemMessage;

public interface BookingAgent {

    @SystemMessage({
            "You are a booking agent for an online hotel. You are here to help customers book rooms and check ",
            "availability. Use the tools you have access to in order to help customers with their requests. You can ",
            "check availability, book rooms, and find bookings."
    })
    String chat(String message);
}
