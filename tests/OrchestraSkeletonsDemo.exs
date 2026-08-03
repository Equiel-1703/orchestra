require Orchestra

Orchestra.defmodule Skeletons do
  include(CAS_Poly)

  defk map_2kernel(a1, a2, a3, size, f) do
    id = get_global_id(0)

    if(id < size) do
      a3[id] = f(a1[id], a2[id])
    end
  end

  defk reduce_kernel(arr, result, initial, f, n) do
    tid = get_global_id(0)
    stride = get_global_size(0)

    temp = initial

    while tid < n do
      temp = f(arr[tid], temp)
      tid = tid + stride
    end

    current_value = result[0]

    while !(current_value == cas_int(result, current_value, f(temp, current_value))) do
      current_value = result[0]
    end
  end

  def map_2(a1, a2, f, %Orchestra.Context{device: :gpu} = ctx) do
    shape = Orchestra.get_shape(a1)
    type = Orchestra.get_type(a1)
    size = Tuple.product(shape)

    res =
      Orchestra.with ctx do
        a1_gnx = Orchestra.new_gnx(a1)
        a2_gnx = Orchestra.new_gnx(a2)
        result_gnx = Orchestra.new_gnx(shape, type)

        Orchestra.spawn(
          &Skeletons.map_2kernel/5,
          {size},
          {64},
          [a1_gnx, a2_gnx, result_gnx, size, f]
        )

        Orchestra.get_gnx(result_gnx)
      end

    res
  end

  def map_2(a1, a2, f, %Orchestra.Context{device: :cpu} = ctx) do
    shape = Orchestra.get_shape(a1)
    type = Orchestra.get_type(a1)
    size = Tuple.product(shape)

    result_tensor = Orchestra.tensor(shape, type)

    Orchestra.with ctx do
      Orchestra.spawn(
        &Skeletons.map_2kernel/5,
        {size},
        {0},
        [a1, a2, result_tensor, size, f]
      )
    end

    result_tensor
  end

  def reduce(arr, initial_value, f, %Orchestra.Context{device: :gpu} = ctx) do
    arr_len = Orchestra.get_shape(arr) |> Tuple.product()
    arr_type = Orchestra.get_type(arr)

    initial_tensor = Orchestra.tensor([initial_value], arr_type)

    block_size = 64
    n_blocks = div(arr_len + block_size - 1, block_size)

    res =
      Orchestra.with ctx do
        arr_gnx = Orchestra.new_gnx(arr)
        result_gnx = Orchestra.new_gnx(initial_tensor)

        Orchestra.spawn(
          &Skeletons.reduce_kernel/5,
          {n_blocks},
          {block_size},
          [arr_gnx, result_gnx, initial_value, f, arr_len]
        )

        Orchestra.get_gnx(result_gnx)
      end

    res[0] |> Nx.to_number()
  end

  def reduce(arr, initial_value, f, %Orchestra.Context{device: :cpu} = ctx) do
    arr_len = Orchestra.get_shape(arr) |> Tuple.product()
    arr_type = Orchestra.get_type(arr)

    result_tensor = Orchestra.tensor([initial_value], arr_type)

    Orchestra.with ctx do
      Orchestra.spawn(
        &Skeletons.reduce_kernel/5,
        {arr_len},
        {0},
        [arr, result_tensor, initial_value, f, arr_len]
      )
    end

    result_tensor[0] |> Nx.to_number()
  end
end

arr_1 = Orchestra.tensor({1024}, :s32, fn _ -> 1 end)
arr_2 = Orchestra.tensor({1024}, :s32, fn _ -> 2 end)

result =
  Skeletons.map_2(arr_1, arr_2, Orchestra.phok(fn x, y -> x * y end), Orchestra.gpu())
  |> Skeletons.reduce(0, Orchestra.phok(fn x, y -> x + y end), Orchestra.cpu())

IO.inspect(result)
