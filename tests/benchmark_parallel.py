"""Benchmark sequential vs parallel PDF processing.
Run: python benchmark_parallel.py
"""
import time
import sys, pathlib
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from app.services.data_processor import get_pdf_processor


def benchmark(parallel: bool):
    processor = get_pdf_processor()
    start = time.perf_counter()
    result = processor.process_all_pdfs(split=True, parallel=parallel)
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    print("Benchmarking PDF processing (sequential vs parallel)\n")
    seq_result, seq_time = benchmark(parallel=False)
    par_result, par_time = benchmark(parallel=True)

    print("Files processed:", len(seq_result["processed_files"]))
    print("Pages:", seq_result["total_pages"])
    print("Chunks (seq/par):", seq_result["total_chunks"], par_result["total_chunks"])
    print("Chunks match:", seq_result["total_chunks"] == par_result["total_chunks"])
    print(f"Sequential time: {seq_time:.3f}s")
    print(f"Parallel time:   {par_time:.3f}s")
    if par_time < seq_time:
        print("\nParallel processing is faster.")
    else:
        print("\nParallel processing not faster for this dataset (may be overhead).")


if __name__ == "__main__":
    main()
