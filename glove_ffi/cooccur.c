#include "cooccur.h"
#include <stdlib.h>

unsigned long read_cooccur(COO_ARR *coo_arr, FILE *f, int lines) {
  CREC *crec = init_crec(lines);
  unsigned long lines_read = fread(crec, sizeof(CREC), lines, f);

  if (lines_read == 0) {
    free(crec);
    return 0;
  }
  int *words_1 = (int *)malloc(lines_read * sizeof(int));
  int *words_2 = (int *)malloc(lines_read * sizeof(int));
  real *vals = (real *)malloc(lines_read * sizeof(real));

  for (int i = 0; i < lines_read; i++) {
    words_1[i] = crec[i].word1;
    words_2[i] = crec[i].word2;
    vals[i] = crec[i].val;
  }

  coo_arr->words_1 = words_1;
  coo_arr->words_2 = words_2;
  coo_arr->vals = vals;

  free(crec);

  return lines_read;
}

void free_cooarr(COO_ARR *coo_arr) {
  free(coo_arr->words_1);
  free(coo_arr->words_2);
  free(coo_arr->vals);
}

CREC *init_crec(int arr_size) {
  CREC *crecs = (CREC *)malloc(arr_size * sizeof(CREC));
  return crecs;
}

FILE *open_file(char *filename) { return fopen(filename, "rb"); }

int close_file(FILE *f) { return fclose(f); }
