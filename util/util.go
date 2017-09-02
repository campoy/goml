package util

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/campoy/tools/imgcat"
	"github.com/gonum/plot"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// ParseMatrix parses a CSV encoded file and returns a matrix of float64 values.
func ParseMatrix(path string) (mat.Matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrapf(err, "could not read %s", path)
	}
	defer f.Close()

	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, errors.Wrap(err, "could not read records")
	}

	data := make([]float64, 0, len(recs)*len(recs[0]))
	for _, rec := range recs {
		for _, v := range rec {
			x, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, errors.Wrapf(err, "could not parse float %s", v)
			}
			data = append(data, x)
		}
	}

	return mat.NewDense(len(recs), len(recs[0]), data), nil
}

// PrintMatrix prints the given matrix to stdout.
func PrintMatrix(name string, m *mat.Dense) {
	fmt.Println(name)
	c, r := m.Dims()
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			fmt.Printf("%.2f\t", m.At(i, j))
		}
		fmt.Println()
	}
}

// PrintPlot prints a plot to the given encoder.
func PrintPlot(enc *imgcat.Encoder, p *plot.Plot) {
	if enc == nil {
		return
	}

	wc := enc.Writer()
	wt, err := p.WriterTo(256, 256, "png")
	if err != nil {
		log.Fatalf("could not create writer from plot: %v", err)
	}
	if _, err := wt.WriteTo(wc); err != nil {
		log.Fatalf("could not write to imgcat: %v", err)
	}
	if err := wc.Close(); err != nil {
		log.Fatalf("could not close imgcat: %v", err)
	}
}
