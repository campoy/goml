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
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

func main() {
	enc, err := imgcat.NewEncoder(os.Stdout,
		imgcat.Width(imgcat.Cells(100)), imgcat.Inline(true))
	if err != nil {
		fmt.Fprintf(os.Stdout, "images will not be shown: %v\n", err)
	}

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

	p := plotDataset(X, y)
	util.PrintPlot(enc, p, 400, 400)

	initialTheta := mat.New(n+1, 1)
	cost, grad := costFunction(initialTheta, X, y)
	fmt.Printf("Cost at initial theta (zeros): %f\n", cost)
	fmt.Printf("Expected cost (approx): 0.693\n")
	fmt.Printf("Gradient at initial theta (zeros): \n")
	for i := 0; i < grad.Cols(); i++ {
		fmt.Printf(" %.4f\n", grad.At(0, i))
	}
	fmt.Printf("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n")

	testTheta := mat.FromSlice(3, 1, []float64{-24, 0.2, 0.2})
	cost, grad = costFunction(testTheta, X, y)

	fmt.Printf("\nCost at test theta: %f\n", cost)
	fmt.Printf("Expected cost (approx): 0.218\n")
	fmt.Printf("Gradient at test theta: \n")
	for i := 0; i < grad.Cols(); i++ {
		fmt.Printf(" %.4f\n", grad.At(0, i))
	}
	fmt.Printf("Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n")

	theta := initialTheta
	for i := 0; true; i++ {
		theta = optimize(func(theta mat.Matrix) (float64, mat.Matrix) {
			return costFunction(theta, X, y)
		}, theta, 250000)
		if i == 0 && i%100 != 0 {
			continue
		}
		acc := accuracy(X, theta, y)
		if acc >= 0.90 {
			break
		}

		fmt.Printf("Train accurracy: %f\n", acc)
		p := plotDataset(X, y)
		p.Add(line(theta))
		p.X.Min, p.X.Max = 0, 100
		p.Y.Min, p.Y.Max = 0, 100
		util.PrintPlot(enc, p, 400, 400)
	}

	cost, _ = costFunction(theta, X, y)
	fmt.Printf("Cost at theta found by optimization: %f\n", cost)
	fmt.Printf("theta: \n")
	fmt.Printf(" %v \n", theta)

	p.Add(line(theta))
	p.X.Min, p.X.Max = 0, 100
	p.Y.Min, p.Y.Max = 0, 100
	util.PrintPlot(enc, p, 400, 400)

	// For a student with scores 45 and 85, we predict an admission probability of 0.776289
	prob := sigmoid(mat.Product(mat.FromSlice(1, 3, []float64{1, 45, 85}), theta).At(0, 0))
	fmt.Printf("For a student with scores 45 and 85, we predict an admission probability of %f\n", prob)
	fmt.Println("Expected value: 0.775 +/- 0.002")

	fmt.Printf("Train accurracy: %f\n", accuracy(X, theta, y))

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

func sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

func optimize(cost func(theta mat.Matrix) (float64, mat.Matrix), initialTheta mat.Matrix, iters int) mat.Matrix {
	theta := initialTheta
	alpha := 0.0001
	for i := 0; i < iters; i++ {
		_, grad := cost(theta)
		theta = mat.Minus(theta, grad.Scale(alpha))
	}
	return theta
}

func accuracy(X, theta, y mat.Matrix) float64 {
	m := X.Rows()
	preds := mat.Map(sigmoid, mat.Product(X, theta))
	correct := 0
	for i := 0; i < m; i++ {
		pred := preds.At(i, 0)
		switch y.At(i, 0) {
		case 0:
			if pred <= 0.5 {
				correct++
			}
		case 1:
			if pred > 0.5 {
				correct++
			}
		}
	}
	return float64(correct) / float64(m)
}

func plotDataset(X, y mat.Matrix) *plot.Plot {
	p, _ := plot.New()
	addScatter := func(val float64, shape draw.GlyphDrawer, color color.Color) *plotter.Scatter {
		values := X.FilterRows(func(i int) bool { return y.At(i, 0) == val })
		s, _ := plotter.NewScatter(xyer.FromMatrixCols(values, 1, 2))
		s.GlyphStyle = draw.GlyphStyle{Shape: shape, Color: color, Radius: 2}
		p.Add(s)
		return s
	}

	pos := addScatter(1, draw.CrossGlyph{}, color.Black)
	neg := addScatter(0, draw.CircleGlyph{}, color.RGBA{255, 255, 0, 255})

	max := func(a, b float64) float64 {
		if a > b {
			return a
		}
		return b
	}
	min := func(a, b float64) float64 {
		if a < b {
			return a
		}
		return b
	}

	p.X.Label.Text = "Exam 1 score"
	p.Y.Label.Text = "Exam 2 score"
	p.Legend.Add("Admitted", pos)
	p.Legend.Add("Not admitted", neg)
	p.Legend.Top = true
	ex1 := X.SliceCols(0, 0)
	ex2 := X.SliceCols(1, 1)
	p.X.Min, p.X.Max = ex1.Reduce(100, min), ex1.Reduce(0, max)
	p.Y.Min, p.Y.Max = ex2.Reduce(100, min), ex2.Reduce(0, max)
	return p
}

func line(theta mat.Matrix) *plotter.Line {
	equation := func(x float64) float64 {
		return -(theta.At(1, 0)*x + theta.At(0, 0)) / theta.At(2, 0)
	}

	line, _, _ := plotter.NewLinePoints(xyer.FromSliceOfSlices([][]float64{
		{0, equation(0)},
		{100, equation(100)},
	}))
	line.Color = color.RGBA{255, 0, 0, 255}
	return line
}
