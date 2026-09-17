package calc

import "testing"

func TestAdd(t *testing.T) {
	c := &Calculator{Base: 10}
	if c.Add(5) != 15 {
		t.Errorf("expected 15")
	}
}
