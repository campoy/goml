package util

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"github.com/campoy/mat"
	"github.com/campoy/tools/imgcat"
	"github.com/pkg/errors"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/vg"
)

// ParseMatrix parses a CSV encoded file and returns a matrix of float64 values.
func ParseMatrix(path string) (mat.Matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return mat.Matrix{}, errors.Wrapf(err, "could not read %s", path)
	}
	defer f.Close()

	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return mat.Matrix{}, errors.Wrap(err, "could not read records")
	}

	data := make([]float64, 0, len(recs)*len(recs[0]))

	for _, rec := range recs {
		for _, v := range rec {
			x, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return mat.Matrix{}, errors.Wrapf(err, "could not parse float %s", v)
			}
			data = append(data, x)
		}
	}

	return mat.FromSlice(len(recs), len(recs[0]), data), nil
}

// PrintPlot prints a plot to the given encoder.
func PrintPlot(enc *imgcat.Encoder, p *plot.Plot, width, height vg.Length) {
	if enc == nil {
		return
	}

	wc := enc.Writer()
	wt, err := p.WriterTo(width, height, "png")
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
