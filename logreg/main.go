package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"os"

	"github.com/campoy/goml/iplot/xyer"
	"github.com/campoy/goml/util"
	"github.com/campoy/mat"
	"github.com/campoy/tools/imgcat"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg/draw"
)

func main() {
	enc, err := imgcat.NewEncoder(os.Stdout,
		imgcat.Width(imgcat.Cells(100)), imgcat.Inline(true))
	if err != nil {
		fmt.Fprintf(os.Stdout, "images will not be shown: %v\n", err)
	}
	_ = enc

	data, err := util.ParseMatrix("ex2data1.txt")
	if err != nil {
		log.Fatalf("could not parse ex2data1.txt: %v", err)
	}

	m, cols := data.Rows(), data.Cols()
	n := cols - 1

	X := mat.ConcatenateCols(mat.New(m, 1).AddScalar(1), data.SliceCols(0, n))
	y := data.SliceCols(n, n+1)

	fmt.Println("Plotting data with + indicating (y=1) examples" +
		"and o indicating (y = 0) examples.")

	{ // plot a line with the costs
		p, _ := plot.New()

		addScatter := func(val float64, shape draw.GlyphDrawer, color color.Color) *plotter.Scatter {
			values := X.FilterRows(func(i int) bool { return y.At(i, 0) == val })
			s, _ := plotter.NewScatter(xyer.FromMatrixCols(values, 1, 2))
			s.GlyphStyle = draw.GlyphStyle{Shape: shape, Color: color}
			p.Add(s)
			return s
		}

		pos := addScatter(1, draw.CrossGlyph{}, color.Black)
		neg := addScatter(0, draw.CircleGlyph{}, color.RGBA{255, 255, 0, 255})

		p.X.Label.Text = "Exam 1 score"
		p.Y.Label.Text = "Exam 2 score"
		p.Legend.Add("Admitted", pos)
		p.Legend.Add("Not admitted", neg)
		p.Legend.Top = true
		util.PrintPlot(enc, p, 400, 400)
	}

	initialTheta := mat.New(n+1, 1)
	cost, grad := costFunction(initialTheta, X, y)
	fmt.Printf("Cost at initial theta (zeros): %f\n", cost)
	fmt.Printf("Expected cost (approx): 0.693\n")
	fmt.Printf("Gradient at initial theta (zeros): \n")
	for i := 0; i < grad.Cols(); i++ {
		fmt.Printf(" %.4f\n", grad.At(0, i))
	}
	fmt.Printf("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n")

	testTheta, err := mat.FromSlice(3, 1, []float64{-24, 0.2, 0.2})
	if err != nil {
		log.Fatal(err)
	}
	cost, grad = costFunction(testTheta, X, y)

	fmt.Printf("\nCost at test theta: %f\n", cost)
	fmt.Printf("Expected cost (approx): 0.218\n")
	fmt.Printf("Gradient at test theta: \n")
	for i := 0; i < grad.Cols(); i++ {
		fmt.Printf(" %.4f\n", grad.At(0, i))
	}
	fmt.Printf("Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n")

	theta := optimize(func(theta mat.Matrix) (float64, mat.Matrix) {
		return costFunction(theta, X, y)
	}, initialTheta, 500000)

	cost, _ = costFunction(theta, X, y)
	fmt.Printf("Cost at theta found by optimization: %f\n", cost)
	fmt.Printf("theta: \n")
	fmt.Printf(" %v \n", theta)

	{ // plot a line with the costs
		p, _ := plot.New()

		addScatter := func(val float64, shape draw.GlyphDrawer, color color.Color) *plotter.Scatter {
			values := X.FilterRows(func(i int) bool { return y.At(i, 0) == val })
			s, _ := plotter.NewScatter(xyer.FromMatrixCols(values, 1, 2))
			s.GlyphStyle = draw.GlyphStyle{Shape: shape, Color: color}
			p.Add(s)
			return s
		}

		pos := addScatter(1, draw.CrossGlyph{}, color.Black)
		neg := addScatter(0, draw.CircleGlyph{}, color.RGBA{255, 255, 0, 255})

		//  plotter.NewLinePoints(xyer.FromSlice{[]float64})

		p.X.Label.Text = "Exam 1 score"
		p.Y.Label.Text = "Exam 2 score"
		p.Legend.Add("Admitted", pos)
		p.Legend.Add("Not admitted", neg)
		p.Legend.Top = true
		util.PrintPlot(enc, p, 400, 400)
	}

}

func costFunction(theta, X, y mat.Matrix) (float64, mat.Matrix) {
	h := mat.Map(sigmoid, mat.Product(X, theta))

	m := float64(X.Rows())
	ones := mat.New(X.Rows(), 1).AddScalar(1)

	j := -1 / m * mat.Sum(mat.Plus(
		mat.Dot(y, mat.Map(math.Log, h)),
		mat.Dot(mat.Minus(ones, y), mat.Map(math.Log, mat.Minus(ones, h))),
	))

	grad := mat.Product(mat.Minus(h, y).T(), X).Scale(1 / m).T()
	return j, grad
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func optimize(cost func(theta mat.Matrix) (float64, mat.Matrix), initialTheta mat.Matrix, iters int) mat.Matrix {
	theta := initialTheta
	prevCost := 1e10
	alpha := 0.01
	for i := 0; i < iters; i++ {
		cost, grad := cost(theta)
		if i%(iters/100) == 0 {
			fmt.Println(cost)
		}
		if prevCost < cost {
			alpha = alpha * 0.9999
		}
		prevCost = cost
		theta = mat.Minus(theta, grad.Scale(alpha))
	}
	return theta
}
