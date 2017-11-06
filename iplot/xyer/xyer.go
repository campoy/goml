package xyer

import (
	"fmt"

	"github.com/campoy/mat"
	"gonum.org/v1/plot/plotter"
)

// // FromMatrices returns an XYer from two matrices.
// func FromMatrices(x, y mat.Matrix) plotter.XYer { return matrices{x, y} }

// type matrices struct{ x, y mat.Matrix }

// func (m matrices) XY(i int) (x, y float64) { return m.x.At(i, 0), m.y.At(i, 0) }
// func (m matrices) Len() int                { s, _ := m.x.Dims(); return s }

func FromMatrixCols(m mat.Matrix, x, y int) plotter.XYer { return matrixCols{m, x, y} }

type matrixCols struct {
	m    mat.Matrix
	x, y int
}

func (m matrixCols) XY(i int) (x, y float64) { return m.m.At(i, m.x), m.m.At(i, m.y) }
func (m matrixCols) Len() int                { return m.m.Rows() }

func FromSlice(x []float64) plotter.XYer { return slice(x) }

type slice []float64

func (s slice) XY(i int) (x, y float64) { return float64(i), s[i] }
func (s slice) Len() int                { return len(s) }

func FromSliceOfSlices(x [][]float64) plotter.XYer {
	if len(x) > 0 && len(x[0]) != 2 {
		panic(fmt.Sprintf("FromSliceOfSlices requires each row to have two elements"))
	}
	return sliceOfSlices(x)
}

type sliceOfSlices [][]float64

func (s sliceOfSlices) XY(i int) (x, y float64) { return s[i][0], s[i][1] }
func (s sliceOfSlices) Len() int                { return len(s) }
