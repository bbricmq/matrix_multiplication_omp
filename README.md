# Parallel matrix multiplication algorithms
#### Elements of the double type were used in range [1.0, 10000.0]
### Computing time
| Size | Naive, 2 threads | Strassen | Fox |
|-|:-:|:-:|:-:|
| 256 | 0.092 s | 0.055 s | 0.02 s |
| 512 | 2.5 s | 2.7 s | 0.87 s |
| 1024 | 12 s | 4.2 s | 3.4 s |
| 2048 | 49.2 s | 35.22 s | 25 s |
| 4096 | 230.25 s | 184.5 s | 225.1 s |

### The amount of memory consumed
| Size | Naive, 2 threads | Strassen | Fox |
|-|:-:|:-:|:-:|
| 256 | 10.47 Mb | 12.12 Mb | 11.61 Mb |
| 512 | 13.61 Mb | 18.02 Mb |  15.3 Mb |
| 1024 | 26.5 Mb | 75.53 Mb | 43.51 Mb |
| 2048 | 75.53 Mb | 170.41 Mb | 139.79 Mb |
| 4096 | 265.61 Mb | 380.4 Mb | 300.32 Mb |

### Acceleration of parallel algorithms for matrix multiplication depending on the number threads
| Number of threads | Naive | Strassen | Fox |
|-|:-:|:-:|:-:|
| 4 | 2.1 s | 3.3 s | 4.2 s |
| 9 | 2.24 s | 2.8 s |  6.5 s |
