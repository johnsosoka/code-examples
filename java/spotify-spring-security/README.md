# Getting Started

This is a simple demonstration of authenticating with the Spotify Oauth 2.0 API Auth Code flow using
a `SecurityFilterChain` 

## Prerequisites

Spotify App Configured

- [Create a Spotify App](https://developer.spotify.com/dashboard/applications)
- Add `http://localhost:8080/login/oauth2/code/spotify` to the Redirect URIs
- Update `application.yml` with the `client-id` and `client-secret` from the Spotify App

## Usage

Start the application and navigate to `http://localhost:8080/greeting` in your web browser.

This should redirect you to the Spotify login page. After logging in, you should be redirected back to the application
and greeted with your Spotify username.