package imgcat

import (
	"encoding/base64"
	"fmt"
	"io"
)

func New(w io.Writer, width string) io.WriteCloser {
	pr, pw := io.Pipe()
	enc := base64.NewEncoder(base64.StdEncoding, pw)

	res := writer{enc, pr, make(chan struct{})}
	go func() {
		fmt.Fprintf(w, "\x1b]1337;File=inline=1;width=%s:", width)
		io.Copy(w, pr)
		fmt.Fprintf(w, "\a\n")
		close(res.done)
	}()

	return res
}

type writer struct {
	io.WriteCloser
	c    io.Closer
	done chan struct{}
}

func (w writer) Close() error {
	w.c.Close()
	<-w.done
	return w.WriteCloser.Close()
}
