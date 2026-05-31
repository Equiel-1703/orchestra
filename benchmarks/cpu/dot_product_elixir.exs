[arg] = System.argv()

size = String.to_integer(arg)

list_a = List.duplicate(1, size)
list_b = List.duplicate(2, size)

prev = System.monotonic_time()

res = Enum.zip_reduce(list_a, list_b, 0, fn a, b, acc ->
  acc + (a * b)
end)

next = System.monotonic_time()

IO.puts("Elixir\t#{size}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
