package com.johnsosoka.springaibooking.service;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.boot.test.context.SpringBootTest;

import java.time.LocalDate;


@SpringBootTest
@ExtendWith(MockitoExtension.class)
class HotelBookingServiceTest {

    private HotelBookingService bookingService;

    @BeforeEach
    void setUp() {
        bookingService = new HotelBookingService();
    }

    @Test
    void testSetAndGetAvailability() {
        LocalDate date = LocalDate.now();
        int numberOfRooms = 5;
        bookingService.setAvailability(date, numberOfRooms);
        assertEquals(numberOfRooms, bookingService.getAvailability(date));
    }

    @Test
    void testIsAvailable_WhenRoomsAvailable() {
        LocalDate date = LocalDate.now();
        bookingService.setAvailability(date, 3);
        assertTrue(bookingService.isAvailable(date));
    }

    @Test
    void testIsAvailable_WhenNoRoomsAvailable() {
        LocalDate date = LocalDate.now();
        bookingService.setAvailability(date, 0);
        assertFalse(bookingService.isAvailable(date));
    }

    @Test
    void testBookRoom_WhenRoomAvailable() {
        LocalDate checkInDate = LocalDate.now();
        LocalDate checkOutDate = checkInDate.plusDays(2);
        String guestName = "John Doe";
        bookingService.setAvailability(checkInDate, 2);

        String result = bookingService.bookRoom(guestName, checkInDate, checkOutDate);
        assertEquals("Room booked successfully for " + guestName, result);
        assertEquals(1, bookingService.getAvailability(checkInDate));
    }

    @Test
    void testBookRoom_WhenNoRoomAvailable() {
        LocalDate checkInDate = LocalDate.now();
        LocalDate checkOutDate = checkInDate.plusDays(2);
        String guestName = "John Doe";
        bookingService.setAvailability(checkInDate, 0);

        String result = bookingService.bookRoom(guestName, checkInDate, checkOutDate);
        assertEquals("No room available for the selected date.", result);
        assertEquals(0, bookingService.getAvailability(checkInDate));
    }

    @Test
    void testCancelBooking_WhenBookingExists() {
        LocalDate checkInDate = LocalDate.now();
        LocalDate checkOutDate = checkInDate.plusDays(2);
        String guestName = "John Doe";
        bookingService.setAvailability(checkInDate, 2);
        bookingService.bookRoom(guestName, checkInDate, checkOutDate);

        String result = bookingService.cancelBooking(guestName);
        assertEquals("Booking canceled for " + guestName, result);
        assertEquals(2, bookingService.getAvailability(checkInDate));
    }

    @Test
    void testCancelBooking_WhenBookingDoesNotExist() {
        String guestName = "John Doe";

        String result = bookingService.cancelBooking(guestName);
        assertEquals("No booking found for " + guestName, result);
    }
}