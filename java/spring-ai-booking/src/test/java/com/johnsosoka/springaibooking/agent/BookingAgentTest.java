package com.johnsosoka.springaibooking.agent;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;


@SpringBootTest
class BookingAgentTest {

    @Autowired
    private BookingAgent bookingAgent;

    @Test
    public void testBookingConversation() {
        String firstMessage = "Hi, my name is John--Can you see if any rooms are available on February 28, 2025?";
        System.out.println(bookingAgent.handleMessage(firstMessage));
        String availability = "Do you have any availability on January 15th, 2025?";
        System.out.println(bookingAgent.handleMessage(availability));
        // Start a new conversation
        String alternativeDate = "Please book 1 room for John on January 15h, 2025. The check-out date will be January 21st, 2025.";
        // Expect a successful booking
        System.out.println(bookingAgent.handleMessage(alternativeDate));
        String checkBooking = "Can you see if a guest John has reserved any rooms?";
        // Expect a yes
        System.out.println(bookingAgent.handleMessage(checkBooking));

        // Demonstrate persisted conversation context
        String summarize = "Can you summarize our discussion today?";
        System.out.println(bookingAgent.handleMessage(summarize));
    }
}