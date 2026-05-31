arg =
  try do
    [arg] = System.argv()
    arg
  rescue
    _ ->
      IO.puts("Usage: mix run benchmarks/mm.ex [MATRIX_SIZE]")
      IO.puts("  - MATRIX_SIZE: Size of the square matrices to be multiplied (MxM)")
      System.halt(0)
  end

size = String.to_integer(arg)

matrix_a = Nx.broadcast(1.0, {size, size}, type: :f32)
matrix_b = Nx.broadcast(2.0, {size, size}, type: :f32)

timing_start = System.monotonic_time()

_result = Nx.dot(matrix_a, matrix_b)

timing_end = System.monotonic_time()

# Calculate times in milliseconds
time = System.convert_time_unit(timing_end - timing_start, :native, :millisecond)

IO.puts("Nx.dot\t#{size}\t#{time}")
