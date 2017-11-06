package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/campoy/goml/mnist/logreg"
	"github.com/campoy/goml/mnist/mnist"
	"github.com/campoy/mat"
	"github.com/campoy/tools/imgcat"
)

func main() {
	imagesPath := flag.String("i", "data/train-images-idx3-ubyte.gz", "path to the file containing all the images")
	labelsPath := flag.String("l", "data/train-labels-idx1-ubyte.gz", "path to the file containing all the labels")
	flag.Parse()

	images, err := mnist.DecodeImages(*imagesPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not decode images: %v\n", err)
		os.Exit(2)
	}
	labels, err := mnist.DecodeLabels(*labelsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not decode labels: %v\n", err)
		os.Exit(2)
	}

	theta := train(images, labels)
	fmt.Println("storing theta:", theta.Rows(), theta.Cols())
}

func train(images [][]byte, labels []byte) mat.Matrix {
	enc, err := imgcat.NewEncoder(os.Stdout, imgcat.Width(imgcat.Percent(25)))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
	}

	for i, img := range images[:10] {
		fmt.Println("label:", labels[i])
		mnist.PlotImage(enc, img)
	}

	fmt.Println("press enter to continue")
	fmt.Scanln()

	m := len(images)
	n := len(images[0])
	k := 10

	x := mat.FromFunc(m, n, func(i, j int) float64 { return float64(images[i][j])/255 + 0.5 })
	x = mat.ConcatenateCols(mat.New(m, 1).AddScalar(1), x)

	// // 😇
	// m = 100
	// x = x.SliceRows(0, m)

	y := mat.FromFunc(m, k, func(i, j int) float64 {
		if labels[i] == byte(j) {
			return 1
		}
		return 0
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	theta := logreg.Fit(ctx, x, y)

	acc, missed := logreg.Accuracy(x, theta, y)
	fmt.Printf("Train accurracy: %f\n", acc)

	fmt.Println("misspredicted")
	for _, i := range missed {
		fmt.Println("label:", labels[i])
		pred := logreg.HotDecode(
			logreg.Predict(x.SliceRows(i, i+1), theta))
		fmt.Println("predicted:", int(pred.At(0, 0)))
		mnist.PlotImage(enc, images[i])
	}

	return mat.Matrix{}
}
