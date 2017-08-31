package iplot

import "testing"

func TestRange(t *testing.T) {
	tc := []struct {
		name  string
		r     Range
		steps int
		vals  map[int]float64
	}{
		{
			name:  "from 0 to 10 one by one",
			r:     Range{Step: 1, From: 0, To: 10},
			steps: 10,
			vals:  map[int]float64{5: 5.0, 0: 0.0, 10: 10.0},
		},
		{
			name:  "from -10 to 10 two by two",
			r:     Range{Step: 2, From: -10, To: 10},
			steps: 10,
			vals:  map[int]float64{0: -10, 5: 0, 10: 10},
		},
		{
			name:  "from -10 to 10 by 0.1 steps",
			r:     Range{Step: 0.1, From: -10, To: 10},
			steps: 200,
			vals:  map[int]float64{0: -10, 50: -5, 100: 0, 150: 5, 200: 10},
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			if s := tt.r.Steps(); s != tt.steps {
				t.Fatalf("expected %d steps; got %d", tt.steps, s)
			}
			for x, wants := range tt.vals {
				if got := tt.r.Map(x); got != wants {
					t.Errorf("expected map(%d) to be %f; got %f", x, wants, got)
				}
			}
		})
	}
}
