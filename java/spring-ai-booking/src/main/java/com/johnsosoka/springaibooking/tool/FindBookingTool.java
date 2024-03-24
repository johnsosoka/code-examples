package com.johnsosoka.springaibooking.tool;


import com.johnsosoka.springaibooking.service.HotelBookingService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.function.Function;

@Component
@RequiredArgsConstructor
public class FindBookingTool implements Function<FindBookingTool.Request, FindBookingTool.Response> {

    private final HotelBookingService hotelBookingService;

    public record Request(String guestName) {}
    public record Response(String booking) {}

    @Override
    public Response apply(Request request) {
        String booking = hotelBookingService.findBookingByGuestNameStr(request.guestName);

        return new Response(booking);
    }

}
