package calc

type Calculator struct {
	Base int
}

func (c *Calculator) Add(x int) int {
	return c.Base + x
}
