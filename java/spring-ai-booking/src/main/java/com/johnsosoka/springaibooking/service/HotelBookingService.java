package com.johnsosoka.springaibooking.service;

import jakarta.annotation.PostConstruct;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service class to manage hotel bookings.
 * All reservations are managed in memory.
 * This class is designed to be used as a singleton bean in a Spring application.
 * It provides methods to set availability, book rooms, cancel bookings, and check availability.
 * The booking data is stored in thread-safe data structures to ensure data consistency across requests.
 */
@Service
@Scope("singleton")
@Slf4j
public class HotelBookingService {
    private Map<LocalDate, Integer> availabilityMap = new ConcurrentHashMap<>();
    private List<Booking> bookings = new ArrayList<>();

    /**
     * Initializes the availability of rooms for specific dates for demonstration purposes.
     */
    @PostConstruct
    public void init() {
        // Set availability for January 15, 2025 (available)
        LocalDate availableDate = LocalDate.of(2025, 1, 15);
        setAvailability(availableDate, 1);

        // Set availability for February 28, 2025 (unavailable)
        LocalDate unavailableDate = LocalDate.of(2025, 2, 28);
        setAvailability(unavailableDate, 0);
    }

    /**
     * Sets the availability of rooms for a specific date.
     *
     * @param date          the date for which to set the availability
     * @param numberOfRooms the number of available rooms for the specified date
     */
    public synchronized void setAvailability(LocalDate date, int numberOfRooms) {
        availabilityMap.put(date, numberOfRooms);
    }

    /**
     * Retrieves the number of available rooms for a specific date.
     *
     * @param date the date for which to retrieve the availability
     * @return the number of available rooms for the specified date
     */
    public int getAvailability(LocalDate date) {
        return availabilityMap.getOrDefault(date, 0);
    }

    /**
     * Checks if there are available rooms for a specific date.
     *
     * @param date the date for which to check the availability
     * @return {@code true} if there are available rooms, {@code false} otherwise
     */
    public synchronized boolean isAvailable(LocalDate date) {
        return getAvailability(date) > 0;
    }

    /**
     * Books a room for a guest with the specified check-in and check-out dates.
     *
     * @param guestName    the name of the guest making the booking
     * @param checkInDate  the check-in date for the booking
     * @param checkOutDate the check-out date for the booking
     * @return a string indicating the status of the booking
     */
    public synchronized String bookRoom(String guestName, LocalDate checkInDate, LocalDate checkOutDate) {
        if (isAvailable(checkInDate)) {
            Booking booking = new Booking(guestName, checkInDate, checkOutDate);
            bookings.add(booking);
            availabilityMap.put(checkInDate, getAvailability(checkInDate) - 1);
            log.info("Room booked successfully for {}", guestName);
            return "Room booked successfully for " + guestName;
        } else {
            log.warn("No room available for {}", guestName);
            return "No room available for the selected date.";
        }
    }

    /**
     * Cancels a booking for a guest with the specified name.
     *
     * @param guestName the name of the guest whose booking should be canceled
     * @return a string indicating the status of the cancellation
     */
    public synchronized String cancelBooking(String guestName) {
        Booking booking = findBookingByGuestName(guestName);
        if (booking != null) {
            bookings.remove(booking);
            availabilityMap.put(booking.getCheckInDate(), getAvailability(booking.getCheckInDate()) + 1);
            log.info("Booking canceled for {}", guestName);
            return "Booking canceled for " + guestName;
        } else {
            log.warn("No booking found for {}", guestName);
            return "No booking found for " + guestName;
        }
    }

    private synchronized Booking findBookingByGuestName(String guestName) {
        return bookings.stream()
                .filter(booking -> booking.getGuestName().equalsIgnoreCase(guestName))
                .findFirst()
                .orElse(null);
    }

    /**
     * Finds a booking by the guest's name and returns booking information as a string.
     *
     * @param guestName the name of the guest to search for
     * @return a string with booking information if found, or a string indicating no booking found
     */
    public synchronized String findBookingByGuestNameStr(String guestName) {
        return bookings.stream()
                .filter(booking -> booking.getGuestName().equalsIgnoreCase(guestName))
                .findFirst()
                .map(booking -> "Booking found for " + booking.getGuestName() +
                        ", Check-in Date: " + booking.getCheckInDate() +
                        ", Check-out Date: " + booking.getCheckOutDate())
                .orElse("No booking found for " + guestName);
    }

    @Data
    @AllArgsConstructor
    private static class Booking {
        private String guestName;
        private LocalDate checkInDate;
        private LocalDate checkOutDate;
    }
}