package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"
	"unicode"

	"github.com/go-chi/chi/v5"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

var s string

func main() {
	fmt.Println("Starting Server......")

	r := chi.NewRouter()

	r.Get("/get", getHandler)
	r.Post("/post", postHandler)
	r.Handle("/", http.FileServer(http.Dir("./public")))

	http.ListenAndServe(":3000", r)
}

func getHandler(w http.ResponseWriter, r *http.Request) {
	s = r.FormValue("query")
	http.Redirect(w, r, "/", 302)
}
func postHandler(w http.ResponseWriter, r *http.Request) {
	res := ""
	keys := [7]string{"(greeting)", "(liked)", "(disliked)", "(food,order,pizza)", "(food,order,hamburger)", "(food,order,salad)", "(food,order,soda)"}
	answer := [7]string{"Hola, en que podemos ayudarte", "Que bueno que te haya gustado", "Lamentamos que no te haya gustado, vamos a revisar",
		"Pizza, buena elección", "Hamburguesa, buena elección", "Ensalada, buena elección", "Gaseosa, buena elección"}
	check := true // variable para revisar se todos los keys se activan
	mayor := float32(0)
	mayor_k := ""
	for i := 0; i < len(keys); i += 1 {
		val_k, bool_k := test(strings.ToLower(s), keys[i])
		fmt.Println(keys[i], val_k, bool_k)
		if bool_k {
			check = check && true
			if val_k > mayor {
				mayor = val_k
				mayor_k = keys[i] + answer[i]
			}
		} else {
			check = check && false
		}
	}
	if check || mayor_k == "" {
		res = "Por favor indique su solicitud"
	} else {
		res = mayor_k
	}
	out_s := `{"in":"` + s + `","out":"` + res + `"}`
	w.Write([]byte(out_s))
	s = ""
}

//////////////////////////////////////////////////////////////////////////////////////////////

func isMn(r rune) bool {
	return unicode.Is(unicode.Mn, r) // Mn: nonspacing marks
}
func test(texto, key string) (float32, bool) {
	content, err := ioutil.ReadFile("training/chats_ref")
	if err != nil {
		log.Fatal(err)
	}
	chats := string(content)
	t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
	normchats, _, _ := transform.String(t, chats)
	palabras := strings.Fields(normchats)

	content, err = ioutil.ReadFile("training/" + key)
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
		if rec_texto(p, texto) {
			matr = append(matr, 1)
		} else {
			matr = append(matr, 0)
		}
	}
	mat_g = append(mat_g, matr)
	matr = nil
	out, _ := multiply(mat_g, w)
	out = sigmoid(out)
	if out[0][0] > 0.8 {
		return out[0][0], true
	} else {
		return out[0][0], false
	}

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
func sigmoid(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = make([]float32, 1)
		out[i][0] = float32(1 / (1 + math.Exp(-float64(x[i][0]))))
	}
	return out
}
func rec_texto(ref, text string) bool {
	s1_array := strings.Fields(ref)
	s2_array := strings.Fields(text)
	for _, i := range s1_array {
		for _, j := range s2_array {
			sum := 0.0
			for _, j_l := range j {
				if strings.Contains(i, string(j_l)) {
					sum = sum + 1
				}
			}
			if len(j) > len(i) {
				sum = sum / float64(len(j))
			} else {
				sum = sum / float64(len(i))
			}
			if sum > 0.65 {
				return true
			}
		}

	}
	return false
}
