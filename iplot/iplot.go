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
