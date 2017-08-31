package main

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/campoy/goml/iplot"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/pkg/errors"
)

func main() {
	// Make a plot and set its title.
	p, err := plot.New()
	if err != nil {
		log.Fatalf("error: %v\n", err)
	}

	xys, err := parseXYs("data.txt")
	if err != nil {
		log.Fatal(err)
	}

	scatter, err := plotter.NewScatter(xys)
	if err != nil {
		log.Fatal(err)
	}
	p.Add(scatter)

	// draw a grid
	p.Add(plotter.NewGrid())
	iplot.Print(os.Stdout, p)
}

func parseXYs(path string) (plotter.XYs, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrapf(err, "could not read %s", path)
	}
	defer f.Close()

	var xys plotter.XYs
	r := csv.NewReader(f)
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errors.Wrap(err, "could not read record")
		}
		x, err := strconv.ParseFloat(rec[0], 64)
		if err != nil {
			return nil, errors.Wrapf(err, "could not parse float %s", rec[0])
		}
		y, err := strconv.ParseFloat(rec[1], 64)
		if err != nil {
			return nil, errors.Wrapf(err, "could not parse float %s", rec[1])
		}
		xys = append(xys, struct{ X, Y float64 }{x, y})
	}

	return xys, nil
}
