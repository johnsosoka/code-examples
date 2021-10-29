package com.johnsosoka.tutorial.demo;

import com.johnsosoka.tutorial.model.bean.BookFormBean;
import com.johnsosoka.tutorial.model.entity.Book;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class SimpleExampleTest {

    private final String bookTitle = "clean code";
    private final String bookIsbn = "9780132350884";

    private SimpleExample simpleExample;

    @Before
    public void setUp() {
        simpleExample = new SimpleExample();
    }

    @Test
    public void testOldObjectMappingStrategyCanMapObjects() {
        BookFormBean sourceFormBean = producePopulatedBookFormBean();
        Book mappedTestObj = simpleExample.oldObjectMappingStrategy(sourceFormBean);

        assertEquals(sourceFormBean.getTitle(), mappedTestObj.getTitle());
        assertEquals(sourceFormBean.getIsbn(), mappedTestObj.getIsbn());
    }

    @Test
    public void testNewObjectMappingStrategyCanMapObjects() {
        BookFormBean sourceFormBean = producePopulatedBookFormBean();
        Book mappedTestObj = simpleExample.newObjectMappingStrategy(sourceFormBean);

        assertEquals(sourceFormBean.getTitle(), mappedTestObj.getTitle());
        assertEquals(sourceFormBean.getIsbn(), mappedTestObj.getIsbn());
    }

    @Test
    public void newMappingStrategyReduced() {
        BookFormBean sourceFormBean = producePopulatedBookFormBean();
        Book mappedTestObj = simpleExample.newMappingStrategyReduced(sourceFormBean);

        assertEquals(sourceFormBean.getTitle(), mappedTestObj.getTitle());
        assertEquals(sourceFormBean.getIsbn(), mappedTestObj.getIsbn());
    }

    /**
     * Helper method to generate a valid BookFormBean for testing
     * @return
     */
    public BookFormBean producePopulatedBookFormBean() {
        BookFormBean bookFormBean = new BookFormBean();
        bookFormBean.setTitle(bookTitle);
        bookFormBean.setIsbn(bookIsbn);
        return bookFormBean;
    }
}
