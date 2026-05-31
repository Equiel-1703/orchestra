defmodule MatrixMath do
  def generate(size) when is_integer(size) and size > 0 do
    Enum.map(1..size, fn _ ->
      Enum.map(1..size, fn _ -> :rand.uniform(100) * 1.0 end)
    end)
  end

  def multiply(matrix_a, matrix_b) do
    # Transpose matrix B so we can iterate over its columns as rows
    matrix_b_t = transpose(matrix_b)

    # Map over each row of A
    Enum.map(matrix_a, fn row_a ->
      # For each row in A, map over the columns of B
      Enum.map(matrix_b_t, fn col_b ->
        dot_product(row_a, col_b)
      end)
    end)
  end

  defp dot_product(list_a, list_b) do
    Enum.zip_reduce(list_a, list_b, 0.0, fn a, b, acc ->
      acc + (a * b)
    end)
  end

  defp transpose(matrix) do
    matrix
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end
end

[arg] = System.argv()

size = String.to_integer(arg)

matrix_a = MatrixMath.generate(size)
matrix_b = MatrixMath.generate(size)

prev = System.monotonic_time()

_res = MatrixMath.multiply(matrix_a, matrix_b)

next = System.monotonic_time()

IO.puts("Elixir\t#{size}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
