package main

import (
	"fmt"
	"log"
	"os"

	"github.com/campoy/goml/iplot"
	"github.com/campoy/goml/parse"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot/plotter"
)

func main() {
	mat, err := parse.Float64Matrix("data.txt")
	if err != nil {
		log.Fatal(err)
	}

	m, _ := mat.Dims()
	X := mat64.NewDense(m, 2, nil)
	X.SetCol(1, mat64.Col(nil, 0, mat))
	for i := 0; i < m; i++ {
		X.Set(i, 0, 1)
	}
	y := mat64.NewDense(m, 1, mat64.Col(nil, 1, mat))

	iplot.Print(os.Stdout, iplot.Scatter(X.ColView(1), y.ColView(0),
		iplot.WithTitle("Population and Profit"),
		iplot.WithXLabel("Population of City in 10,000s"),
		iplot.WithYLabel("Profit in $10,000s"),
	))

	theta := mat64.NewDense(2, 1, make([]float64, 2))
	cost := computeCost(X, y, theta)
	fmt.Printf("initial cost is %f\n", cost)

	theta, costs := gradientDescent(X, y, theta, 0.01, 1500)
	fmt.Printf("after 1500 steps of gradient descent with learning rate 0.01\n")
	fmt.Printf("theta: %v\n", theta.RawMatrix().Data)

	cost = computeCost(X, y, theta)
	fmt.Printf("current cost is %f\n", cost)

	iplot.Print(os.Stdout, iplot.Line(costs,
		iplot.WithTitle("cost over time"),
		iplot.WithXLabel("number of iterations"),
	))

	pred := new(mat64.Dense)
	pred.Mul(mat64.NewDense(1, 2, []float64{1, 3.5}), theta)
	fmt.Printf("prediction for a city 35,000 inhabitants is %f\n", 10000*pred.At(0, 0))

	pred.Mul(mat64.NewDense(1, 2, []float64{1, 7}), theta)
	fmt.Printf("prediction for a city 70,000 inhabitants is %f\n", 10000*pred.At(0, 0))

	p := iplot.Scatter(X.ColView(1), y.ColView(0),
		iplot.WithTitle("Population and Profit"),
		iplot.WithXLabel("Population of City in 10,000s"),
		iplot.WithYLabel("Profit in $10,000s"),
	)

	p.Add(plotter.NewFunction(func(x float64) float64 {
		return theta.At(0, 0) + theta.At(1, 0)*x
	}))
	iplot.Print(os.Stdout, p)
}

// computeCost computes the cost of using theta as the parameter for linear
// regression to fit the data points in X and y.
func computeCost(X, y, theta *mat64.Dense) float64 {
	m, _ := X.Dims()
	h := new(mat64.Dense)
	h.Mul(X, theta)
	h.Sub(h, y)
	h.Apply(func(i, j int, v float64) float64 { return v * v }, h)
	return mat64.Sum(h) / float64(2*m)
}

// gradientDescent performs gradient descent to learn theta returns the updated
// theta after taking iters gradient steps with learning rate alpha.
func gradientDescent(X, y, theta *mat64.Dense, alpha float64, iters int) (*mat64.Dense, []float64) {
	costs := make([]float64, 0, iters)

	m, _ := X.Dims()
	for i := 0; i < iters; i++ {
		h := new(mat64.Dense)
		h.Mul(X, theta) // (X * theta - y) * X
		h.Sub(h, y)

		grad := new(mat64.Dense) // -alpha/m (h . X)
		grad.Mul(X.T(), h)
		grad.Scale(alpha/float64(m), grad)

		theta.Sub(theta, grad)

		costs = append(costs, computeCost(X, y, theta))
	}

	return theta, costs
}
