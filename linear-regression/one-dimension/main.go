package main

import (
	"bytes"
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
	"github.com/campoy/goml/parse"
)

func main() {
	rand.Seed(time.Now().Unix())

	enc, err := imgcat.NewEncoder(os.Stdout, imgcat.Width(imgcat.Cells(100)), imgcat.Inline(true))
	if err != nil {
		log.Fatal(err)
	}

	data, err := parse.Float64Matrix("data.txt")
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
		print(enc, p)
	}

	theta := mat.NewDense(2, 1, make([]float64, 2))

	cost := computeCost(X, y, theta)
	fmt.Printf("initial cost is %f\n", cost)

	alpha := 0.01
	epsilon := 1e-8
	theta, thetas, costs := gradientDescent(X, y, theta, alpha, epsilon)
	fmt.Printf("took %d steps\n", len(thetas))
	fmt.Printf("theta: %v\n", theta.RawMatrix().Data)
	fmt.Printf("current cost is %f\n", computeCost(X, y, theta))

	{ // plot a line with the costs
		p, _ := plot.New()
		l, _ := plotter.NewLine(xyer.FromSlice(costs))
		p.Add(l)
		l.Color = color.RGBA{R: 255, A: 255}
		p.Title.Text = "Cost over time"
		p.X.Label.Text = "number of iterations"
		print(enc, p)
	}

	{ // plot optimization space
		fmt.Println("optimization space and traject")
		p, _ := plot.New()
		p.Add(plotter.NewContour(
			iplot.GridXYZ(iplot.NewRange(-8, 5, 100), iplot.NewRange(0, 3, 100),
				func(a, b float64) float64 {
					return computeCost(X, y, mat.NewDense(2, 1, []float64{a, b}))
				}),
			nil, palette.Heat(16, 1)))
		s, _ := plotter.NewScatter(xyer.FromSliceOfSlices(thetas))
		s.Radius = 1
		s.Shape = draw.CrossGlyph{}
		p.Add(s)
		p.Title.Text = "Optimization path"
		p.X.Label.Text = "theta0"
		p.Y.Label.Text = "theta1"
		print(enc, p)
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
		print(enc, p)
	}
}

// computeCost computes the cost of using theta as the parameter for linear
// regression to fit the data points in X and y.
func computeCost(X, y, theta *mat.Dense) float64 {
	m, _ := X.Dims()
	h := new(mat.Dense)
	h.Mul(X, theta)
	h.Sub(h, y)
	h.Apply(func(i, j int, v float64) float64 { return v * v }, h)
	return mat.Sum(h) / float64(2*m)
}

// gradientDescent performs gradient descent to learn theta returns the updated
// theta after taking iters gradient steps with learning rate alpha.
func gradientDescent(X, y, theta *mat.Dense, alpha, epsilon float64) (*mat.Dense, [][]float64, []float64) {
	var costs []float64
	var thetas [][]float64

	m, _ := X.Dims()
	for len(costs) < 2 || costs[len(costs)-2]-costs[len(costs)-1] > epsilon {
		h := new(mat.Dense)
		h.Mul(X, theta) // (X * theta - y) * X
		h.Sub(h, y)

		grad := new(mat.Dense) // -alpha/m (h . X)
		grad.Mul(X.T(), h)
		grad.Scale(alpha/float64(m), grad)
		theta.Sub(theta, grad)

		costs = append(costs, computeCost(X, y, theta))
		thetas = append(thetas, []float64{theta.At(0, 0), theta.At(1, 0)})
	}

	return theta, thetas, costs
}

func print(enc *imgcat.Encoder, p *plot.Plot) {
	wt, err := p.WriterTo(512, 256, "png")
	if err != nil {
		log.Fatalf("could not print plot: %v", err)
	}
	buf := new(bytes.Buffer)
	if _, err := wt.WriteTo(buf); err != nil {
		log.Fatalf("could not write to buffer: %v", err)
	}
	if err := enc.Encode(buf); err != nil {
		log.Fatalf("could not encode: %v", err)
	}
	fmt.Println("press enter to continue ...")
	fmt.Scanln()
}
