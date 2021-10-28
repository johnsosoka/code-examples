package com.johnsosoka.tutorial.model.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ArticleDTO {

    private String authorFirstName;
    private String authorLastName;
    private String articleTitle;
    private String articleContent;

}
