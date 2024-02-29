package com.johnsosoka.spotispring.service;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.stereotype.Component;

/**
 * Service class to handle greeting logic
 */
@Component
public class GreetingService {

    public String greetAuthenticatedUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        DefaultOAuth2User user = (DefaultOAuth2User) authentication.getPrincipal();
        String responseMessage = "Hello, " + user.getName() + "!";

        return responseMessage;
    }


}
