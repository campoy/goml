package iplot

import (
	"encoding/base64"
	"fmt"
	"image/color"
	"io"
	"log"
	"os/exec"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg/draw"
)

// An Option provides a way to customize a plot.
type Option func(p *plot.Plot)

// WithXLabel customizes the label for the X axis in a plot.
func WithXLabel(text string) Option {
	return func(p *plot.Plot) { p.X.Label.Text = text }
}

// WithYLabel customizes the label for the X axis in a plot.
func WithYLabel(text string) Option {
	return func(p *plot.Plot) { p.Y.Label.Text = text }
}

// WithTitle customizes the title in a plot.
func WithTitle(text string) Option {
	return func(p *plot.Plot) { p.Title.Text = text }
}

// Line prints a scatter graph of the given points versus
// their position in the vector.
func Line(y []float64, options ...Option) *plot.Plot {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}

	var xys plotter.XYs
	for i := range y {
		xys = append(xys, struct{ X, Y float64 }{float64(i), y[i]})
	}
	line, err := plotter.NewLine(xys)
	if err != nil {
		log.Fatal(err)
	}
	line.Color = color.RGBA{R: 255, A: 255}
	p.Add(line)
	p.Add(plotter.NewGrid())

	for _, option := range options {
		option(p)
	}
	return p
}

// A Range represents a range of values with steps.
type Range struct {
	From, To float64
	Step     float64
}

func NewRange(from, to float64, steps int) Range {
	return Range{
		From: from,
		To:   to,
		Step: (to - from) / float64(steps),
	}
}

// Steps returns how many total steps there are in between From and To.
func (r Range) Steps() int { return int((r.To - r.From) / r.Step) }

// Map returns the value of the nth step between From and To.
func (r Range) Map(n int) float64 { return r.From + float64(n)*r.Step }

// Grid3D can be used as a plotter.XYZer for HeatMap and Countour.
type Grid3D struct {
	XRange Range
	YRange Range
	F      func(x, y float64) float64
}

func (g Grid3D) Dims() (int, int)   { return g.XRange.Steps(), g.YRange.Steps() }
func (g Grid3D) X(c int) float64    { return g.XRange.Map(c) }
func (g Grid3D) Y(c int) float64    { return g.YRange.Map(c) }
func (g Grid3D) Z(c, r int) float64 { return g.F(g.X(c), g.Y(r)) }

// Scatter prints a scatter graph of the given points
func Scatter(x, y *mat64.Vector, options ...Option) *plot.Plot {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	if x.Len() != y.Len() {
		panic("dimensions of x and y are different")
	}

	var xys plotter.XYs
	for i := 0; i < x.Len(); i++ {
		xys = append(xys, struct{ X, Y float64 }{x.At(i, 0), y.At(i, 0)})
	}
	scatter, err := plotter.NewScatter(xys)
	if err != nil {
		log.Fatal(err)
	}
	scatter.Color = color.RGBA{R: 255, A: 255}
	scatter.Shape = draw.CrossGlyph{}
	p.Add(scatter)
	p.Add(plotter.NewGrid())

	for _, option := range options {
		option(p)
	}
	return p
}

// Print prints the given plot into an iTerm writer.
func Print(out io.Writer, p *plot.Plot) {
	w, err := p.WriterTo(256, 256, "png")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\x1b]1337;File=inline=1;width=50%%:")
	enc := base64.NewEncoder(base64.StdEncoding, out)
	if _, err := w.WriteTo(enc); err != nil {
		log.Fatal(err)
	}
	if err := enc.Close(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\a\n")
}

// Clear clears the given iTerm writer.
func Clear(out io.Writer) {
	cmd := exec.Command("clear")
	cmd.Stdout = out
	if err := cmd.Run(); err != nil {
		panic(err)
	}
}
