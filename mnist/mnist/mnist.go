package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"

	"github.com/campoy/tools/imgcat"
	"github.com/pkg/errors"
)

func DecodeImages(path string) ([][]byte, error) {
	r, err := openGz(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var header struct{ Magic, N, Rows, Cols int32 }
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, errors.New("bad header")
	}
	if header.Magic != 2051 {
		return nil, errors.New("wrong magic number in header")
	}

	bytes := make([]byte, header.N*header.Rows*header.Cols)
	if _, err = io.ReadFull(r, bytes); err != nil {
		return nil, errors.Wrap(err, "could not read full")
	}

	return split(bytes, int(header.Rows*header.Cols)), nil
}

func DecodeLabels(path string) ([]byte, error) {
	r, err := openGz(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var header struct{ Magic, N int32 }
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, errors.New("bad header")
	}
	if header.Magic != 2049 {
		return nil, errors.New("wrong magic number in header")
	}

	bytes := make([]byte, header.N)
	if _, err = io.ReadFull(r, bytes); err != nil {
		return nil, errors.Wrap(err, "could not read full")
	}
	return bytes, nil
}

func openGz(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrapf(err, "could not open images file %q", path)
	}
	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, errors.Wrapf(err, "could not unzip images file %q", path)
	}
	return r, nil
}

func split(b []byte, n int) [][]byte {
	s := make([][]byte, len(b)/n)
	for i := range s {
		s[i] = b[i*n : (i+1)*n : (i+1)*n]
	}
	return s
}

func PlotImage(enc *imgcat.Encoder, b []byte) error {
	l := int(math.Sqrt(float64(len(b))))
	m := image.NewGray(image.Rect(0, 0, l, l))
	for i := 0; i < l; i++ {
		for j := 0; j < l; j++ {
			m.SetGray(i, j, color.Gray{b[i+j*l]})
		}
	}

	w := enc.Writer()
	defer w.Close()
	return png.Encode(w, m)
}
