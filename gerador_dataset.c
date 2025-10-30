#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Gera um arquivo de texto com um dataset de pontos com coordenadas inteiras.
 *
 * Este programa cria um arquivo contendo M pontos em um espaço D-dimensional,
 * com coordenadas inteiras aleatórias no intervalo [0, max_val].
 */
int main(int argc, char* argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Uso: %s <num_pontos> <num_dimensoes> <max_val> <arquivo_saida>\n", argv[0]);
    fprintf(stderr, "Exemplo: %s 1000000 10 10000 dataset.txt\n", argv[0]);
    return EXIT_FAILURE;
  }

  int num_points = atoi(argv[1]);
  int num_dimensions = atoi(argv[2]);
  int max_val = atoi(argv[3]);
  const char* output_filename = argv[4];

  if (num_points <= 0 || num_dimensions <= 0 || max_val <= 0) {
    fprintf(stderr, "Erro: O número de pontos, dimensões e o valor máximo devem ser positivos.\n");
    return EXIT_FAILURE;
  }

  FILE* file = fopen(output_filename, "w");
  if (file == NULL) {
    perror("Erro ao abrir o arquivo de saída");
    return EXIT_FAILURE;
  }

  srand(time(NULL));

  printf("Gerando '%s' com %d pontos, %d dimensões e valores até %d...\n",
         output_filename, num_points, num_dimensions, max_val);

  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_dimensions; j++) {
      // Gera um inteiro aleatório no intervalo [0, max_val]
      int random_val = rand() % (max_val + 1);
      fprintf(file, "%d%c", random_val, (j == num_dimensions - 1) ? '\n' : ' ');
    }
  }

  fclose(file);
  printf("Dataset gerado com sucesso!\n");

  return EXIT_SUCCESS;
}