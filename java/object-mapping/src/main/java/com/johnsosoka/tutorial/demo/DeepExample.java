package com.johnsosoka.tutorial.demo;

import com.johnsosoka.tutorial.model.dto.ArticleDTO;
import com.johnsosoka.tutorial.model.entity.Article;
import com.johnsosoka.tutorial.model.entity.Author;
import org.modelmapper.ModelMapper;

public class DeepExample {

    // TODO mirror SimpleExample structure?
    private void moreAdvancedMapping() {
        Author authorObj = new Author("john", "smith");
        Article article = new Article();
        article.setTitle("My interesting title");
        article.setContent("...the intriguing contents");
        article.setAuthor(authorObj);

        // Map it
        ArticleDTO mappedObj = new ModelMapper().map(article, ArticleDTO.class );

        System.out.println(mappedObj.getAuthorFirstName());
        System.out.println(mappedObj.getAuthorLastName());
        System.out.println(mappedObj.getArticleTitle());
        System.out.println(mappedObj.getArticleContent());
    }


}
