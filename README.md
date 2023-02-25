# Parallel matrix multiplication algorithms
#### Elements of the int type were used in range [1, 10000]
### Computing time
| Size | Naive | Strassen | Fox |
|-|:-:|:-:|:-:|
| 256 | 0.029 s | 0.03 s | 0.011 s |
| 512 | 0.54 s | 0.24 s | 0.23 s |
| 1024 | 4.04 s | 1.21 s | 2.34 s |
| 2048 | 77.3 s | 8.2 s | 49.68 s |
| 4096 | 621 s | 58 s | 421 s |

### The amount of memory consumed
| Size | Naive | Strassen | Fox |
|-|:-:|:-:|:-:|
| 256 | 10.47 Mb | 12.12 Mb | 11.61 Mb |
| 512 | 13.61 Mb | 18.02 Mb |  15.3 Mb |
| 1024 | 26.5 Mb | 75.53 Mb | 43.51 Mb |
| 2048 | 75.53 Mb | 170.41 Mb | 139.79 Mb |
| 4096 | 265.61 Mb | 380.4 Mb | 300.32 Mb |

### Acceleration of parallel algorithms for matrix multiplication depending on the number threads
| Number of threads | Naive | Strassen | Fox |
|-|:-:|:-:|:-:|
| 4 | 2.14 | 2.5 | 5.1 |
| 9 | 2.6 | 2.71 |  7.2 |
