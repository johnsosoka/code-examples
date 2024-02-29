package com.johnsosoka.spotispring.controller;

import com.johnsosoka.spotispring.service.GreetingService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
@RequiredArgsConstructor
public class GreetingController {

    private final GreetingService greetingService;

    @GetMapping("/greeting")
    public @ResponseBody String greeting() {
        return greetingService.greetAuthenticatedUser();
    }
}
