package com.johnsosoka.tutorial.model.entity;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Article {

    private String title;
    private String content;
    private Author author;

}
