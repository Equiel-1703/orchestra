require Orchestra

Orchestra.defmodule Atomics_2 do
  defk atomic_kernel_2(atomic(counter), out_array, size) do
    tid = get_global_id(0)

    if tid < size do
      previous_val = add_atomic_int(counter, 1)

      out_array[tid] = previous_val
    end
  end

  def launch(counter_tensor, total_threads) do
    block_size = 256
    num_blocks = div(total_threads + block_size - 1, block_size)

    {res_counter, res_array} =
      Orchestra.with Orchestra.gpu() do
        counter_gnx = Orchestra.new_gnx(counter_tensor)
        out_array_gnx = Orchestra.new_gnx({total_threads}, :s32)

        Orchestra.spawn(
          &Atomics_2.atomic_kernel_2/3,
          {num_blocks},
          {block_size},
          [counter_gnx, out_array_gnx, total_threads]
        )

        res_counter = Orchestra.get_gnx(counter_gnx)
        res_array = Orchestra.get_gnx(out_array_gnx)

        {res_counter, res_array}
      end

    {res_counter, res_array}
  end
end

# Total threads
total_threads = 1024

# Counter tensor
counter_tensor = Orchestra.tensor([0], :s32)

# Launching kernel
{counter_res, array_res} = Atomics_2.launch(counter_tensor, total_threads)

IO.inspect(counter_res, label: "* Counter")
IO.inspect(array_res, limit: :infinity, label: "* Output Array")
