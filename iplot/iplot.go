package iplot

import (
	"github.com/gonum/plot/plotter"
)

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

// grid3D can be used as a plotter.XYZer for HeatMap and Countour.
type grid3D struct {
	x Range
	y Range
	f func(x, y float64) float64
}

// GridXYZ returns a plot.GridXYZ where each value is obtained by calling
// the given function.
func GridXYZ(x, y Range, f func(x, y float64) float64) plotter.GridXYZ {
	return grid3D{x, y, f}
}

func (g grid3D) Dims() (int, int)   { return g.x.Steps(), g.y.Steps() }
func (g grid3D) X(c int) float64    { return g.x.Map(c) }
func (g grid3D) Y(c int) float64    { return g.y.Map(c) }
func (g grid3D) Z(c, r int) float64 { return g.f(g.X(c), g.Y(r)) }
