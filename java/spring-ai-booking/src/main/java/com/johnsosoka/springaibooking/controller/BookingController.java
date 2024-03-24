package com.johnsosoka.springaibooking.controller;

import com.johnsosoka.springaibooking.agent.BookingAgent;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

/**
 * Controller exposes the HotelBookingAgent via a REST API.
 * See the postman collection for an example request.
 */
@RestController
@RequiredArgsConstructor
public class BookingController {

    private final BookingAgent bookingAgent;

    @PostMapping("/ai/booking")
    public Map<String, String> completion(@RequestBody String message) {
        String response = bookingAgent.handleMessage(message);
        return Map.of("AgentResponse", response);
    }
}