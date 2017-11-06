package logreg

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/campoy/mat"
)

var matProduct = mat.Product

// Accuracy computes the accuracy of x and theta predicting y.
// It also returns a list of the misspredicted rows.
func Accuracy(x, theta, y mat.Matrix) (float64, []int) {
	m := x.Rows()
	preds := HotDecode(Predict(x, theta))
	labels := HotDecode(y)
	correct := 0
	var missed []int
	for i := 0; i < m; i++ {
		if preds.At(i, 0) == labels.At(i, 0) {
			correct++
		} else {
			missed = append(missed, i)
		}
	}
	return float64(correct) / float64(m), missed
}

// Predict computes the prediction of labels given x and theta.
func Predict(x, theta mat.Matrix) mat.Matrix {
	return mat.Map(sigmoid, matProduct(x, theta))
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

func HotEncode(m mat.Matrix, k int) mat.Matrix {
	return mat.FromFunc(m.Rows(), k, func(i, j int) float64 {
		if int(m.At(i, 0)) == j {
			return 1.0
		}
		return 0.0
	})
}

func HotDecode(m mat.Matrix) mat.Matrix {
	return mat.FromFunc(m.Rows(), 1, func(i, j int) float64 {
		v := 0.0
		pos := 0
		for j := 0; j < m.Cols(); j++ {
			if x := m.At(i, j); x > v {
				v = x
				pos = j
			}
		}
		return float64(pos)
	})
}

func Fit(ctx context.Context, x, y mat.Matrix) mat.Matrix {
	start := time.Now()

	initialTheta := mat.New(x.Cols(), y.Cols())

	theta := initialTheta

	for {
		acc, _ := Accuracy(x, theta, y)
		fmt.Printf("t: %v |  accurracy: %f\n", time.Since(start), acc)

		if acc == 1.0 {
			return theta
		}

		select {
		case <-ctx.Done():
			return theta
		default:
		}

		theta = optimize(func(theta mat.Matrix) (float64, mat.Matrix) {
			return costFunction(theta, x, y)
		}, theta, 1)
	}
}

func costFunction(theta, x, y mat.Matrix) (float64, mat.Matrix) {
	h := Predict(x, theta)

	m := float64(x.Rows())
	ones := mat.New(x.Rows(), y.Cols()).AddScalar(1)

	j := -1 / m * mat.Sum(mat.Plus(
		mat.Dot(y, mat.Map(math.Log, h)),
		mat.Dot(mat.Minus(ones, y), mat.Map(math.Log, mat.Minus(ones, h))),
	))

	grad := matProduct(mat.Minus(h, y).T(), x).Scale(1 / m).T()
	return j, grad
}

func optimize(cost func(theta mat.Matrix) (float64, mat.Matrix), initialTheta mat.Matrix, iters int) mat.Matrix {
	theta := initialTheta
	alpha := 0.01
	for i := 0; i < iters; i++ {
		_, grad := cost(theta)
		theta = mat.Minus(theta, grad.Scale(alpha))
	}
	return theta
}
