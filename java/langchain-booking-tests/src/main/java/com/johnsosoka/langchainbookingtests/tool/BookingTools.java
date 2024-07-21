package com.johnsosoka.langchainbookingtests.tool;

import dev.langchain4j.agent.tool.Tool;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import com.johnsosoka.langchainbookingtests.service.HotelBookingService;

import java.time.LocalDate;

@Component
@RequiredArgsConstructor
public class BookingTools {

    private final HotelBookingService hotelBookingService;


    @Tool("Check Availability -- Useful for seeing if a room is available for a given date.")
    public boolean checkAvailability(String date) {
        LocalDate parsedDate = LocalDate.parse(date);
        return hotelBookingService.isAvailable(parsedDate);
    }

    @Tool("Book Room -- Useful for booking a room for a given guest name, check-in date, and check-out date.")
    public String bookRoom(String guestName, String checkInDate, String checkOutDate) {
        LocalDate checkIn = LocalDate.parse(checkInDate);
        LocalDate checkOut = LocalDate.parse(checkOutDate);
        return hotelBookingService.bookRoom(guestName, checkIn, checkOut);
    }

    @Tool("Find Booking -- Useful for finding a booking by guest name.")
    public String findBooking(String guestName) {
        return hotelBookingService.findBookingByGuestNameStr(guestName);
    }

}
