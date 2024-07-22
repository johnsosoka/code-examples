package com.johnsosoka.langchainbookingtests.service;

import com.johnsosoka.langchainbookingtests.agent.BookingAgent;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class ChatService {

    private final BookingAgent bookingAgent;

    public String chat(String message) {
        return bookingAgent.chat(message);
    }

}
