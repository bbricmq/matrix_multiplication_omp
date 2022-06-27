#include <omp.h>
#include "fox.h"
#include "cannon.h"
#include "strassen.h"
#include <windows.h>
#include <stdio.h>
#include <psapi.h>
#include "pdh.h"
#pragma comment(lib, "Pdh.lib")
using namespace std;
long GetPrivateWorkingSet(const wchar_t* process) {
  PDH_HQUERY query;
  PDH_HCOUNTER counter;
  PDH_FMT_COUNTERVALUE counterVal;
  const wchar_t* name = L"\\Process(%s)\\Working Set - Private";
  wchar_t buf[1024] = L"";
  swprintf_s(buf, 1024, name, process);

  PdhOpenQuery(NULL, NULL, &query);
  PDH_STATUS res = PdhAddEnglishCounter(query, buf, NULL, &counter);
  res = PdhCollectQueryData(query);
  PdhGetFormattedCounterValue(counter, PDH_FMT_LONG, NULL, &counterVal);
  long ret = counterVal.longValue;
  PdhCloseQuery(query);

  return ret;
}

void PrintMemory() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc,
                       sizeof(pmc));
  printf("Private committed memory: %u B\nWorking set: %u B\n",
         (UINT)pmc.PrivateUsage, (UINT)pmc.WorkingSetSize);

  printf("Private working set: %u B\n", (UINT)GetPrivateWorkingSet(L"Program"));

  HANDLE heaps[20];
  HEAP_SUMMARY hs;

  DWORD cHeaps = GetProcessHeaps(20, heaps);

  for (DWORD i = 0; i < cHeaps; i++) {
    memset(&hs, 0, sizeof(hs));
    hs.cb = sizeof(hs);
    if (HeapSummary(heaps[i], 0, &hs) == FALSE) printf("HeapSummary failed");
    printf("Heap #%u size: %u B\n", (UINT)i, (UINT)hs.cbAllocated);
  }
  printf("\n");
}

Matrix parallelLentMatrixMultiplication(const std::vector<double>& A,
                                        const std::vector<double>& B,
                                        size_t Size) {
  if (Size <= 0) throw "Block size of matrix must be > 0";
  if ((A.size() <= 0) || (B.size() <= 0)) throw "Size of matrix must be > 0";
  if (A.size() != B.size()) throw "Different size of matrix";

  size_t blocks_num = 2;
  Matrix result(Size, 0);
  omp_set_num_threads(blocks_num * blocks_num);
#pragma omp parallel
  {
    size_t thread_num = omp_get_thread_num();
    size_t block_rows_size = thread_num / blocks_num;
    size_t block_cols_size = thread_num % blocks_num;
    size_t block_size = Size / blocks_num;
    for (size_t i = 0; i < block_size; i++)
      for (size_t j = 0; j < block_size; j++)
        for (size_t k = 0; k < Size; k++)
          result[(block_rows_size * block_size + i) * Size +
                 (block_cols_size * block_size + j)] +=
              A[(block_rows_size * block_size + i) * Size + k] *
              B[k * Size + (block_cols_size * block_size + j)];
  }
  return result;
}

int main() {
  {

      {
      size_t size = 4096;
      Matrix A = createRandomMatrix(size * size);
      Matrix B = createRandomMatrix(size * size);
      cout << "==============================================" << endl;
      cout << "SIZE MATRIX: " << size << endl;
      omp_set_num_threads(4);
      PrintMemory();
      PrintMemory();
      Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
      PrintMemory();
      Matrix C_naivee = strassenMultiplication(A, B, size);
      PrintMemory();
      Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
      PrintMemory();
      omp_set_num_threads(1);
      Matrix C_Cannon = parallelCannonMult(A, B, 4);
      PrintMemory();
      cout << "==============================================" << endl;
    }
  }
  /*
  {
    size_t size = 100;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(4);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }
  {
    size_t size = 200;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 300;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }


    {
    size_t size = 400;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 500;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 600;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 700;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 800;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 900;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1000;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1100;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1200;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1300;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1400;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1500;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1600;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1700;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1800;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }

    {
    size_t size = 1900;
    Matrix A = createRandomMatrix(size * size);
    Matrix B = createRandomMatrix(size * size);
    cout << "==============================================" << endl;
    cout << "SIZE MATRIX: " << size << endl;
    omp_set_num_threads(1);
    PrintMemory();
    PrintMemory();
    Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
    PrintMemory();
    Matrix C_naivee = strassenMultiplication(A, B, size);
    PrintMemory();
    Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    PrintMemory();
    omp_set_num_threads(1);
    Matrix C_Cannon = parallelCannonMult(A, B, 4);
    PrintMemory();
    cout << "==============================================" << endl;
  }
    {
      size_t size = 2000;
      Matrix A = createRandomMatrix(size * size);
      Matrix B = createRandomMatrix(size * size);
      cout << "==============================================" << endl;
      cout << "SIZE MATRIX: " << size << endl;
      omp_set_num_threads(1);
      PrintMemory();
      PrintMemory();
      Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
      PrintMemory();
      Matrix C_naivee = strassenMultiplication(A, B, size);
      PrintMemory();
      Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
      PrintMemory();
      omp_set_num_threads(1);
      Matrix C_Cannon = parallelCannonMult(A, B, 4);
      PrintMemory();
      cout << "==============================================" << endl;
    }


  /*
  omp_set_num_threads(4);
  double t1 = omp_get_wtime();
  Matrix C = parallelBlockMatrixMultiplication(A, B, size * size);
  double t2 = omp_get_wtime();
  std::cout << "Omp Fox 4 threads time: " << t2 - t1 << std::endl;

  omp_set_num_threads(1);
  double t3 = omp_get_wtime();
  Matrix C_block = parallelBlockMatrixMultiplication(A, B, size * size);
  double t4 = omp_get_wtime();
  std::cout << "Seq Fox time: " << t4 - t3 << std::endl;
  
  double t5 = omp_get_wtime();
  Matrix C_naive = sequentialMatrixMultiplication(A, B, size * size);
  double t6 = omp_get_wtime();
  std::cout << "Naive time: " << t6 - t5 << std::endl;
  
  double t7 = omp_get_wtime();
  Matrix C_naivee = strassenMultiplication_omp(A, B, size);
  double t8 = omp_get_wtime();
  std::cout << "Strassen: " << t8 - t7 << std::endl;
  */
  return 0;

 

}

