package parse

import (
	"encoding/csv"
	"os"
	"strconv"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// Float64Matrix parses a CSV encoded file and returns a matrix of float64 values.
func Float64Matrix(path string) (mat.Matrix, error) {
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
