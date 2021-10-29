package com.johnsosoka.tutorial.demo;

import com.johnsosoka.tutorial.model.bean.BookFormBean;
import com.johnsosoka.tutorial.model.entity.Book;
import lombok.NoArgsConstructor;
import org.modelmapper.ModelMapper;

@NoArgsConstructor
public class SimpleExample {

    public void executeSimpleExample() {
        // Create source Object...
        BookFormBean sourceBean = new BookFormBean();
        sourceBean.setTitle("Clean Code");
        sourceBean.setIsbn("9780132350884");

        // Map to new object
        Book destinationObject = this.newMappingStrategyReduced(sourceBean);

        // Validate
        System.out.println(destinationObject.getTitle());
        System.out.println(destinationObject.getIsbn());
    }

    public Book oldObjectMappingStrategy(BookFormBean bookFormBean) {
        Book book = new Book();

        book.setTitle(bookFormBean.getTitle());
        book.setIsbn(bookFormBean.getIsbn());

        return book;
    }

    public Book newObjectMappingStrategy(BookFormBean sourceObject) {
        ModelMapper modelMapper = new ModelMapper();
        Book destinationObject = modelMapper.map(sourceObject, Book.class);
        return destinationObject;
    }

    public Book newMappingStrategyReduced(BookFormBean bookFormBean) {
        return new ModelMapper().map(bookFormBean, Book.class);
    }

}
