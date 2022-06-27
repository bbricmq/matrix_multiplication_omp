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

int main() {
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

