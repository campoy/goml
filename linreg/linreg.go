package linreg

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	mm "github.com/campoy/mat"
)

// ComputeCost computes the cost of using theta as the parameter for linear
// regression to fit the data points in X and y.
func ComputeCost(X, y, theta *mat.Dense) float64 {
	m, _ := X.Dims()
	h := new(mat.Dense)
	h.Mul(X, theta)
	h.Sub(h, y)
	h.MulElem(h, h)
	return mat.Sum(h) / float64(2*m)
}

func ComputeCost(X, y, theta mm.Matrix) float64 {
	h := mm.Minus(mm.Product(X, theta), y)
	return mm.Product(h, h).Sum()
}

// GradientDescent performs gradient descent to learn theta returns the updated
// theta after taking iters gradient steps with learning rate alpha.
func GradientDescent(X, y, theta *mat.Dense, alpha float64, iters int) (*mat.Dense, [][]float64, []float64) {
	var costs []float64
	var thetas [][]float64

	m, _ := X.Dims()
	for i := 0; i < iters; i++ {
		h := new(mat.Dense)
		h.Mul(X, theta) // (X * theta - y) * X
		h.Sub(h, y)

		grad := new(mat.Dense) // -alpha/m (h . X)
		grad.Mul(X.T(), h)
		grad.Scale(alpha/float64(m), grad)
		theta.Sub(theta, grad)

		costs = append(costs, ComputeCost(X, y, theta))
		thetas = append(thetas, []float64{theta.At(0, 0), theta.At(1, 0)})
	}

	return theta, thetas, costs
}

// InitParameters returns the X, y and theta parameters extracted
// from the given data matrix. It uses the last column for y,
// the rest for X, and zeros for theta.
func InitParameters(data mat.Matrix) (X, y, theta *mat.Dense) {
	m, n := data.Dims()
	X = mat.NewDense(m, n, nil)
	for i := 1; i < n; i++ {
		X.SetCol(i, mat.Col(nil, i-1, data))
	}
	for i := 0; i < m; i++ {
		X.Set(i, 0, 1)
	}
	y = mat.NewDense(m, 1, mat.Col(nil, n-1, data))
	theta = mat.NewDense(n, 1, make([]float64, n))
	return X, y, theta
}

// NormalizeFeatures normalizes the given matrix of features and
// returns the matrix of means and standard deviations.
func NormalizeFeatures(X *mat.Dense) (means, stdDevs *mat.Dense) {
	m, n := X.Dims()
	means = mat.NewDense(1, n, nil)
	stdDevs = mat.NewDense(1, n, nil)
	stdDevs.Set(0, 0, 1)

	for col := 1; col < n; col++ {
		mean, stdDev := stat.MeanStdDev(mat.Col(nil, col, X), nil)
		means.Set(0, col, mean)
		stdDevs.Set(0, col, stdDev)
		for i := 0; i < m; i++ {
			X.Set(i, col, (X.At(i, col)-mean)/stdDev)
		}
	}
	return means, stdDevs
}
