package logreg

import (
	"fmt"
	"testing"

	"github.com/campoy/mat"
)

func TestHotDecode(t *testing.T) {
	m := mat.FromSlice(4, 3, []float64{
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		0, 1, 0,
	})
	fmt.Println(HotDecode(m))
}

func TestHotEncode(t *testing.T) {
	m := mat.FromSlice(4, 1, []float64{2, 1, 0, 1})
	fmt.Println(HotEncode(m, 3))
}
