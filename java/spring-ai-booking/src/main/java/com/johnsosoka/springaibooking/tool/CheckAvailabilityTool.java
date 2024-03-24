package com.johnsosoka.springaibooking.tool;

import com.johnsosoka.springaibooking.service.HotelBookingService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.util.function.Function;

@Component
@RequiredArgsConstructor
public class CheckAvailabilityTool implements Function<CheckAvailabilityTool.Request, CheckAvailabilityTool.Response> {

    private final HotelBookingService hotelBookingService;

    public record Request(String date) {}
    public record Response(boolean available) {}

    @Override
    public Response apply(Request request) {
        // LocalDate from a string
        LocalDate date = LocalDate.parse(request.date);
        Boolean isAvailable = hotelBookingService.isAvailable(date);

        return new Response(isAvailable);
    }

}
