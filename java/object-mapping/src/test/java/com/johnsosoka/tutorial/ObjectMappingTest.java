package com.johnsosoka.tutorial;

import org.junit.Test;

public class ObjectMappingTest {

    @Test(expected = Test.None.class )
    public void testCanRunWithoutException() {
        ObjectMapping objectMappingTest = new ObjectMapping();
        objectMappingTest.run();
    }
}
