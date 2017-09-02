package main

import (
	"fmt"
	"image/color"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/campoy/tools/imgcat"
	"github.com/gonum/plot"
	"github.com/gonum/plot/palette"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg/draw"
	"gonum.org/v1/gonum/mat"

	"github.com/campoy/goml/iplot"
	"github.com/campoy/goml/iplot/xyer"
	"github.com/campoy/goml/linreg"
	"github.com/campoy/goml/util"
)

func main() {
	rand.Seed(time.Now().Unix())

	enc, err := imgcat.NewEncoder(os.Stdout,
		imgcat.Width(imgcat.Cells(100)), imgcat.Inline(true))
	if err != nil {
		log.Fatal(err)
	}

	data, err := util.ParseMatrix("data.txt")
	if err != nil {
		log.Fatal(err)
	}

	m, _ := data.Dims()
	X := mat.NewDense(m, 2, nil)
	X.SetCol(1, mat.Col(nil, 0, data))
	for i := 0; i < m; i++ {
		X.Set(i, 0, 1)
	}
	y := mat.NewDense(m, 1, mat.Col(nil, 1, data))

	{ // plot scatter of the points
		fmt.Println("Dataset plot")
		p, _ := plot.New()
		s, _ := plotter.NewScatter(xyer.FromMatrices(X.ColView(1), y))
		p.Add(s)
		s.Color = color.RGBA{R: 255, A: 255}
		s.Shape = draw.CrossGlyph{}
		p.Title.Text = "Population and Profit"
		p.X.Label.Text = "Population of City in 10,000s"
		p.Y.Label.Text = "Profit in $10,000s"
		util.PrintPlot(enc, p)
	}

	theta := mat.NewDense(2, 1, make([]float64, 2))

	cost := linreg.ComputeCost(X, y, theta)
	fmt.Printf("initial cost is %f\n", cost)

	alpha := 0.01
	theta, thetas, costs := linreg.GradientDescent(X, y, theta, alpha, 1500)
	fmt.Printf("took %d steps\n", len(thetas))
	util.PrintMatrix("theta", theta)
	fmt.Printf("current cost is %f\n", linreg.ComputeCost(X, y, theta))

	{ // plot a line with the costs
		p, _ := plot.New()
		l, _ := plotter.NewLine(xyer.FromSlice(costs))
		p.Add(l)
		l.Color = color.RGBA{R: 255, A: 255}
		p.Title.Text = "Cost over time"
		p.X.Label.Text = "number of iterations"
		util.PrintPlot(enc, p)
	}

	{ // plot optimization space
		fmt.Println("optimization space and traject")
		p, _ := plot.New()
		p.Add(plotter.NewContour(
			iplot.GridXYZ(iplot.NewRange(-8, 5, 100), iplot.NewRange(0, 3, 100),
				func(a, b float64) float64 {
					return linreg.ComputeCost(X, y, mat.NewDense(2, 1, []float64{a, b}))
				}),
			nil, palette.Heat(16, 1)))
		s, _ := plotter.NewScatter(xyer.FromSliceOfSlices(thetas))
		s.Radius = 1
		s.Shape = draw.CrossGlyph{}
		p.Add(s)
		p.Title.Text = "Optimization path"
		p.X.Label.Text = "theta0"
		p.Y.Label.Text = "theta1"
		util.PrintPlot(enc, p)
	}

	pred := new(mat.Dense)
	pred.Mul(mat.NewDense(1, 2, []float64{1, 3.5}), theta)
	fmt.Printf("prediction for a city 35,000 inhabitants is %f\n", 10000*pred.At(0, 0))

	pred.Mul(mat.NewDense(1, 2, []float64{1, 7}), theta)
	fmt.Printf("prediction for a city 70,000 inhabitants is %f\n", 10000*pred.At(0, 0))

	{ // print points and prediction model
		fmt.Println("points and prediction model")
		p, _ := plot.New()
		s, _ := plotter.NewScatter(xyer.FromMatrices(X.ColView(1), y))
		s.Color = color.RGBA{R: 255, A: 255}
		s.Shape = draw.CrossGlyph{}
		p.Add(s)
		p.Title.Text = "Population and Profit"
		p.X.Label.Text = "Population of City in 10,000s"
		p.Y.Label.Text = "Profit in $10,000s"
		p.Add(plotter.NewFunction(func(x float64) float64 {
			return theta.At(0, 0) + theta.At(1, 0)*x
		}))
		util.PrintPlot(enc, p)
	}
}
