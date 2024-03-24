package com.johnsosoka.springaibooking.tool;

import com.johnsosoka.springaibooking.service.HotelBookingService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.util.function.Function;

@Component
@RequiredArgsConstructor
public class BookRoomTool implements Function<BookRoomTool.Request, BookRoomTool.Response> {

    private final HotelBookingService hotelBookingService;

    public record Request(String guestName, String checkInDate, String checkOutDate) {}
    public record Response(String bookingStatus) {}

    @Override
    public Response apply(Request request) {
        LocalDate checkIn = LocalDate.parse(request.checkInDate);
        LocalDate checkOut = LocalDate.parse(request.checkOutDate);
        String bookingStatus = hotelBookingService.bookRoom(request.guestName, checkIn, checkOut);

        return new Response(bookingStatus);
    }

}