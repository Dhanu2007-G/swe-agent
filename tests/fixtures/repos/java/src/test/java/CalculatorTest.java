package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class CalculatorTest {
    @Test
    public void testMultiply() {
        Calculator calc = new Calculator();
        assertEquals(10, calc.multiply(2, 5));
    }
}
