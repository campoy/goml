package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"os"

	"github.com/campoy/goml/iplot/xyer"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"

	"github.com/campoy/goml/util"
	"github.com/campoy/tools/imgcat"
	"github.com/gonum/plot/vg/draw"
	"gonum.org/v1/gonum/mat"
)

func main() {
	enc, err := imgcat.NewEncoder(os.Stdout,
		imgcat.Width(imgcat.Cells(100)), imgcat.Inline(true))
	if err != nil {
		fmt.Fprintf(os.Stdout, "images will not be shown: %v", err)
	}

	data, err := util.ParseMatrix("ex2data1.txt")
	if err != nil {
		log.Fatalf("could not parse ex2data1.txt: %v", err)
	}

	m, n := data.Dims()
	n = n - 1 // y is not a feature

	X := mat.NewDense(m, n+1, nil)
	for i := 0; i < m; i++ {
		X.Set(i, 0, 0)
	}
	for i := 0; i < n; i++ {
		X.SetCol(i+1, mat.Col(nil, i, data))
	}
	y := mat.NewDense(m, 1, mat.Col(nil, n, data))

	fmt.Println("Plotting data with + indicating (y=1) examples" +
		"and o indicating (y = 0) examples.")

	{ // plot a line with the costs
		p, _ := plot.New()

		npos := int(mat.Sum(y))
		pos := mat.NewDense(npos, n, nil)
		neg := mat.NewDense(m-npos, n, nil)
		var ipos, ineg int
		for i := 0; i < m; i++ {
			if y.At(i, 0) == 0 {
				neg.SetRow(ineg, X.RawRowView(i)[1:])
				ineg++
			} else {
				pos.SetRow(ipos, X.RawRowView(i)[1:])
				ipos++
			}
		}

		spos, _ := plotter.NewScatter(xyer.FromMatrices(pos.ColView(0), pos.ColView(1)))
		spos.GlyphStyle.Shape = draw.CrossGlyph{}
		spos.GlyphStyle.Color = color.Black
		p.Add(spos)
		sneg, _ := plotter.NewScatter(xyer.FromMatrices(neg.ColView(0), neg.ColView(1)))
		sneg.GlyphStyle.Shape = draw.CircleGlyph{}
		sneg.GlyphStyle.Color = color.RGBA{255, 255, 0, 255}
		p.Add(sneg)

		p.X.Label.Text = "Exam 1 score"
		p.Y.Label.Text = "Exam 2 score"
		p.Legend.Add("Admitted", spos)
		p.Legend.Add("Not admitted", sneg)
		p.Legend.Top = true
		util.PrintPlot(enc, p, 400, 400)
	}

	initialTheta := mat.NewDense(n+1, 1, nil)
	cost, grad := costFunction(initialTheta, X, y)
	fmt.Printf("Cost at initial theta (zeros): %f\n", cost)
	fmt.Printf("Expected cost (approx): 0.693\n")
	fmt.Printf("Gradient at initial theta (zeros): \n")
	fmt.Printf(" %v \n", grad)
	fmt.Printf("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n")
}

func costFunction(theta, X, y *mat.Dense) (float64, []float64) {
	h := new(mat.Dense)
	h.Mul(theta.T(), X.T())
	// fmt.Println("after mult", h)
	h.Apply(wrapApply(sigmoid), h)
	// fmt.Println("after sigmoid", h)

	s1 := mat.DenseCopyOf(y)
	s1.Scale(-1, s1)
	logH := new(mat.Dense)
	logH.Apply(wrapApply(math.Log), h)
	p1 := new(mat.Dense)
	p1.Mul(s1.T(), logH.T())
	fmt.Println(p1)

	s2 := mat.DenseCopyOf(y)
	s2.Apply(wrapApply(func(x float64) float64 { return 1 - x }), s2)
	fmt.Println("s2", s2)
	logOneMinusH := new(mat.Dense)
	logOneMinusH.Apply(wrapApply(func(x float64) float64 { return math.Log(1 - x) }), h)
	fmt.Println("logOneMinusH", logOneMinusH)
	p2 := new(mat.Dense)
	p2.Mul(s2.T(), logOneMinusH.T())
	fmt.Println(p2)

	m, _ := X.Dims()
	fmt.Println(m)
	j := 1 / float64(m) * (mat.Sum(p1) - mat.Sum(p2))

	return j, nil
}

func sigmoid(z float64) float64 {
	return 1 / (1 / math.Exp(-z))
}

func wrapApply(f func(float64) float64) func(i, j int, x float64) float64 {
	return func(i, j int, x float64) float64 {
		return f(x)
	}
}
