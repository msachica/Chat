package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

func isMn(r rune) bool {
	return unicode.Is(unicode.Mn, r) // Mn: nonspacing marks
}
func main() {
	fmt.Println("Hola mundo")
	content, err := ioutil.ReadFile("chats")
	if err != nil {
		log.Fatal(err)
	}
	chats := string(content)
	t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
	normchats, _, _ := transform.String(t, chats)
	palabras_k := strings.Split(normchats, "#")

	keys := [7]string{"(greeting)", "(liked)", "(disliked)", "(food,order,pizza)", "(food,order,hamburger)", "(food,order,salad)", "(food,order,soda)"}

	for _, k := range keys {
		var key_train []string //matriz de palabras de acuerdo con key para entrenamiento
		for _, palabra := range palabras_k {
			if strings.Contains(palabra, k) {
				palabra = strings.ReplaceAll(palabra, k, "")
				key_train = append(key_train, palabra)
			}
		}
		fmt.Println(key_train)

		content, err = ioutil.ReadFile("chats_ref")
		if err != nil {
			log.Fatal(err)
		}

		chats = string(content)
		t = transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
		normchats, _, _ = transform.String(t, chats)
		palabras := strings.Fields(normchats)

		var matr []float32
		var mat_g [][]float32
		for _, palabra := range key_train {
			for _, p := range palabras {
				if strings.Contains(p, palabra) || strings.Contains(palabra, p) {
					matr = append(matr, 1)
				} else {
					matr = append(matr, 0)
				}
			}
			mat_g = append(mat_g, matr)
			matr = nil
		}
		random := rand.New(rand.NewSource(1))

		var weigths []float32
		random.Float32()
		for range palabras {
			weigths = append(weigths, 2*rand.Float32()-1)
		}
		var w [][]float32
		w = transpose(append(w, weigths))

		for i := 0; i < 10000; i += 1 {
			out, _ := multiply(mat_g, w)
			out2, error_tr := sigmoid(out)
			var sum float32 = 0
			w = new_weigths(mat_g, out2, error_tr, w)
			if i%1000 == 0 {
				for j := 0; j < len(error_tr); j += 1 {
					sum = sum + error_tr[j][0]
				}
				error_mean := sum / float32(len(error_tr))
				fmt.Println(k, "Error ", error_mean, " para iteración ", i)
			}
		}

		//fmt.Println(w)

		var var_t []string
		for i := 0; i < len(w); i += 1 {
			s := fmt.Sprintf("%f", w[i][0])
			var_t = append(var_t, s)
		}
		message := []byte(strings.Join(var_t, " "))

		err := ioutil.WriteFile(k, message, 0644)
		if err != nil {
			log.Fatal(err)
		}
	}

	fmt.Println(test("quiero una hamburguesa", "(food,order,hamburger)"))
}

func transpose(x [][]float32) [][]float32 {
	out := make([][]float32, len(x[0]))
	for i := 0; i < len(x); i += 1 {
		for j := 0; j < len(x[0]); j += 1 {
			out[j] = append(out[j], x[i][j])
		}
	}
	return out
}

func multiply(x, y [][]float32) ([][]float32, error) {
	if len(x[0]) != len(y) {
		return nil, errors.New("Can't do matrix multiplication.")
	}

	out := make([][]float32, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = make([]float32, len(y[0]))
		for j := 0; j < len(y[0]); j++ {
			for k := 0; k < len(y); k++ {
				out[i][j] += x[i][k] * y[k][j]
			}
		}
	}
	return out, nil
}
func sigmoid(x [][]float32) ([][]float32, [][]float32) {
	out := make([][]float32, len(x))
	out2 := make([][]float32, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = make([]float32, 1)
		out2[i] = make([]float32, 1)
		out[i][0] = float32(1 / (1 + math.Exp(-float64(x[i][0]))))
		out2[i][0] = 1 - out[i][0]
	}
	return out, out2
}
func new_weigths(x, y, e, w [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = make([]float32, 1)
		out[i][0] = e[i][0] * y[i][0] * (1 - y[i][0])
	}
	adj, _ := multiply(transpose(x), out)
	out2 := make([][]float32, len(adj))
	for i := 0; i < len(adj); i++ {
		out2[i] = make([]float32, 1)
		out2[i][0] = adj[i][0] + w[i][0]
	}
	return out2
}
func test(texto, key string) bool {
	content, err := ioutil.ReadFile("chats_ref")
	if err != nil {
		log.Fatal(err)
	}
	chats := string(content)
	t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
	normchats, _, _ := transform.String(t, chats)
	palabras := strings.Fields(normchats)

	content, err = ioutil.ReadFile(key)
	if err != nil {
		log.Fatal(err)
	}
	weigths := string(content)
	w_str := strings.Fields(weigths)
	w := make([][]float32, len(w_str))
	for i := 0; i < len(w_str); i++ {
		w[i] = make([]float32, 1)
		var_t, _ := strconv.ParseFloat(w_str[i], 32)
		w[i][0] = float32(var_t)
	}

	texto, _, _ = transform.String(t, texto)

	var matr []float32
	var mat_g [][]float32
	for _, p := range palabras {
		if strings.Contains(p, texto) || strings.Contains(texto, p) {
			matr = append(matr, 1)
		} else {
			matr = append(matr, 0)
		}
	}
	mat_g = append(mat_g, matr)
	matr = nil
	out, _ := multiply(mat_g, w)
	out, _ = sigmoid(out)
	fmt.Println(out)
	if out[0][0] > 0.5 {
		return true
	} else {
		return false
	}
}
