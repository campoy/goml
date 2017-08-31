package main

import (
	"log"
	"math"
	"os"

	"github.com/campoy/goml/iplot"
	"github.com/campoy/goml/parse"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/palette"
	"github.com/gonum/plot/plotter"
)

type grid struct{ mat64.Matrix }

func (g grid) Z(c, r int) float64 { return g.At(c, r) }
func (g grid) X(c int) float64    { return float64(c) }
func (g grid) Y(c int) float64    { return float64(c) }

func main() {
	_, err := parse.Float64Matrix("data.txt")
	if err != nil {
		log.Fatal(err)
	}

	data := mat64.NewDense(100, 100, nil)
	c, r := data.Dims()
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			data.Set(i, j, math.Abs(float64((c/2-j)*(r/2-i))))
		}
	}
	g := grid{data}

	plot, err := plot.New()
	p := palette.Heat(16, 1)
	plot.Add(plotter.NewHeatMap(g, p))
	// plot.Add(plotter.NewContour(g, nil, p))
	iplot.Print(os.Stdout, plot)
}
