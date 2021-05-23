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
	//lee chats de entrenamiento (keys) y remueve tildes
	content, err := ioutil.ReadFile("chats")
	if err != nil {
		log.Fatal(err)
	}
	chats := string(content)
	t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
	normchats, _, _ := transform.String(t, chats)
	//separa frases entre #
	palabras_k := strings.Split(normchats, "#")

	keys := [7]string{"(greeting)", "(liked)", "(disliked)", "(food,order,pizza)", "(food,order,hamburger)", "(food,order,salad)", "(food,order,soda)"}

	//Agrega al arreglo key_train[] las frases que corresponden a todos los key y les quita la etiqueta (key)
	//out_train[] es la matriz de entrenamiento de salida (1 o 0) de acuerdo con el key
	for _, k := range keys {
		var key_train []string  //matriz de palabras para entrenamiento de las keys
		var out_train []float32 //matriz de salida de entrenamiento
		for _, palabra := range palabras_k {
			if palabra != "" {
				if strings.Contains(palabra, k) {
					out_train = append(out_train, 1)
				} else {
					out_train = append(out_train, 0)
				}
				palabra = strings.ReplaceAll(palabra, k, "")
				key_train = append(key_train, palabra)
			}
		}
		fmt.Println(key_train)

		//Carga el archivo de palabras diferentes para compararlo con cada una de las frases de entrenamiento
		content, err = ioutil.ReadFile("chats_ref")
		if err != nil {
			log.Fatal(err)
		}
		chats = string(content)
		t = transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
		normchats, _, _ = transform.String(t, chats)
		palabras := strings.Fields(normchats) //genera un arreglo de palabras
		//Compara cada una de las palabras de referencia con las palabras de la frase de entrenamiento
		//Se genera una matriz de 1 y 0 de acuerdo si la palabra de ref está contenida en las frases
		var matr []float32
		var mat_g [][]float32
		for _, palabra := range key_train { //key_train es el arreglo de frases de acuerdo con el key
			for _, p := range palabras { //palabras es el arreglo de palabras de referencia
				if strings.Contains(p, palabra) || strings.Contains(palabra, p) {
					matr = append(matr, 1)
				} else {
					matr = append(matr, 0)
				}
			}
			mat_g = append(mat_g, matr)
			matr = nil
		}
		//Se genera una matriz de los pesos iniciales de la red neuronal, de acuerdo al num de palabras de ref
		random := rand.New(rand.NewSource(1))
		var weigths []float32
		random.Float32()
		for range palabras {
			weigths = append(weigths, 2*rand.Float32()-1)
		}
		var w [][]float32
		w = transpose(append(w, weigths))
		//Genera varias iteraciones de entrenamiento para modificar los pesos y disminuir el error del sistema
		for i := 0; i < 10000; i += 1 {
			out, _ := multiply(mat_g, w)
			out2 := sigmoid(out)

			//encuentra los errores de la salida de la sigmoide respecto a la matriz de salida de entrenamiento
			error_tr := make([][]float32, len(out))
			for m := 0; m < len(out); m++ {
				error_tr[m] = make([]float32, 1)
				error_tr[m][0] = out_train[m] - out2[m][0]
			}

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
		//Genera arreglo de pesos y guarda archivo llamado con la (key)
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

	fmt.Println(test("hola", "(greeting)"))
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
	out = sigmoid(out)
	fmt.Println(out)
	if out[0][0] > 0.5 {
		return true
	} else {
		return false
	}
}
