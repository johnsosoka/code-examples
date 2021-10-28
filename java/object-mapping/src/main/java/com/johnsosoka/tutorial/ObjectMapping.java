package com.johnsosoka.tutorial;

import com.johnsosoka.tutorial.model.dto.ArticleDTO;
import com.johnsosoka.tutorial.model.dto.BookDTO;
import com.johnsosoka.tutorial.model.entity.Article;
import com.johnsosoka.tutorial.model.entity.Author;
import com.johnsosoka.tutorial.model.entity.Book;
import org.modelmapper.ModelMapper;



public class ObjectMapping {



    public static void main(String[] args) {
        ObjectMapping exampleTransform = new ObjectMapping();
        exampleTransform.run();
    }

    // TODO create lesson package, one class for each example? tests?
    public void run() {
        String bookTitle = "Clean Code";
        String bookISBN = "9780132350884";

        // Create source Object...
        BookDTO sourceObject = new BookDTO();
        sourceObject.setTitle(bookTitle);
        sourceObject.setIsbn(bookISBN);

        // Map source object to new object type.
        Book destinationObject = newWay(sourceObject);

        assert(destinationObject.getTitle().equals(bookTitle));
        assert(destinationObject.getIsbn().equals(bookISBN));

        System.out.println(destinationObject.getTitle());
        System.out.println(destinationObject.getIsbn());

        moreAdvancedMapping();
    }

    private Book oldWay(BookDTO bookDTO) {
        Book book = new Book();

        book.setTitle(bookDTO.getTitle());
        book.setIsbn(bookDTO.getIsbn());

        return book;
    }

    private Book newWay(BookDTO bookDTO) {
        return new ModelMapper().map(bookDTO, Book.class);
    }


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
