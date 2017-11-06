package main

import (
	"fmt"
	"image/color"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/campoy/goml/iplot/xyer"
	"github.com/campoy/goml/linreg"
	"github.com/campoy/goml/util"
	"github.com/campoy/tools/imgcat"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

func main() {
	rand.Seed(time.Now().Unix())

	enc, err := imgcat.NewEncoder(os.Stdout,
		imgcat.Width(imgcat.Cells(100)), imgcat.Inline(true))
	if err != nil {
		fmt.Fprintf(os.Stdout, "images will not be shown: %v", err)
	}

	data, err := util.ParseMatrix("data.txt")
	if err != nil {
		log.Fatal(err)
	}

	X, y, theta := linreg.InitParameters(data)
	fmt.Println("First 10 examples from the dataset")
	for i := 0; i < 10; i++ {
		fmt.Printf("x = %v, y = %v\n", X.RawRowView(i)[1:], y.At(i, 0))
	}

	means, stdDevs := linreg.NormalizeFeatures(X)
	fmt.Println("First 10 examples from the dataset after normalization")
	for i := 0; i < 10; i++ {
		fmt.Printf("x = %v, y = %v\n", X.RawRowView(i)[1:], y.At(i, 0))
	}

	fmt.Println("Running gradient descent")
	theta, _, costs := linreg.GradientDescent(X, y, theta, 0.01, 400)

	{ // plot a line with the costs
		p, _ := plot.New()
		l, _ := plotter.NewLine(xyer.FromSlice(costs))
		p.Add(l)
		l.Color = color.RGBA{R: 255, A: 255}
		p.Title.Text = "Cost over time"
		p.X.Label.Text = "number of iterations"
		util.PrintPlot(enc, p)
	}

	house := mat.NewDense(1, 3, []float64{1, 1650, 3})
	// Apply normalization
	house.Sub(house, means)
	house.DivElem(house, stdDevs)
	util.PrintMatrix("features after normalization", house)

	pred := new(mat.Dense)
	pred.Mul(house, theta)
	fmt.Println("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):")
	fmt.Printf("\t%.2f\n", pred.At(0, 0))
}
