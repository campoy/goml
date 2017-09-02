package main

import (
	"image"
	"log"
	"math"
	"os"

	"image/color"
	"image/gif"

	"github.com/campoy/tools/imgcat"
)

const width, height = 128, 128

func cos(x int) int {
	const speed = 1 / (2 * math.Pi / width)
	c := math.Cos(float64(x) / speed)
	return int(height * (1 + c) / 2)
}

func main() {
	palette := color.Palette{color.White, color.Black}

	var g gif.GIF
	for i := 0; i < width; i += 2 {
		m := image.NewPaletted(image.Rect(0, 0, width, height), palette)
		for x := 0; x < m.Bounds().Dx(); x++ {
			m.Set(x, cos(x+i), color.Black)
		}
		g.Image = append(g.Image, m)
		g.Delay = append(g.Delay, 5)
	}

	enc, err := imgcat.NewEncoder(os.Stdout, imgcat.Inline(true), imgcat.Width(imgcat.Percent(50)))
	if err != nil {
		log.Fatal(err)
	}
	wc := enc.Writer()
	if err := gif.EncodeAll(wc, &g); err != nil {
		log.Fatal(err)
	}
	if err := wc.Close(); err != nil {
		log.Fatal(err)
	}
}
