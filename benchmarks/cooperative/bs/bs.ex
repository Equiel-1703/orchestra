require Orchestra

Orchestra.defmodule BezierSurface do
  defd bezier_blend(k, mu, n) do
    nn = 0
    kn = 0
    nkn = 0

    blend = 1.0
    nn  = n
    kn  = k
    nkn = n - k

    while nn >= 1 do
      blend = blend * nn
      nn = nn - 1
      if kn > 1 do
        blend = blend / (kn * 1.0)
        kn = kn - 1
      end
      if nkn > 1 do
        blend = blend / (nkn * 1.0)
        nkn = nkn - 1
      end
    end

    if k > 0 do
      blend = blend * pow(mu, k)
    end

    if n - k > 0 do
      blend = blend * pow(1 - mu, n - k)
    end

    return blend
  end

  defk bezier_surface_cpu(
    xyz_in, xyz_outp,
    n_tasks, alpha,
    n_work_items,
    in_size_i, in_size_j,
    out_size_i, out_size_j
  ) do
    n_threads = get_global_size(0)
    tid = get_global_id(0)

    cut = n_tasks
    if alpha >= 0.0 && alpha <= 1.0 do
      cut = n_tasks * alpha
    end

    wg_in_J = (out_size_j - 1) / n_work_items + 1
    wg_in_I = (out_size_i - 1) / n_work_items + 1

    current = tid
    t = current

    while current < cut do
      my_s1 = t / wg_in_J
      my_s0 = t
      while my_s0 >= wg_in_J do
        my_s0 = my_s0 - wg_in_J
      end

      row = my_s1 * n_work_items
      col = my_s0 * n_work_items

      i = row
      while i < row + n_work_items do
        mui = i / (1.0 * (out_size_i - 1))
        j = col
        while j < col + n_work_items do
          muj = j / (1.0 * (out_size_j - 1))
          if i < out_size_i && j < out_size_j do
            out[3]
            out[0] = 0.0
            out[1] = 0.0
            out[2] = 0.0
            ki = 0
            while ki <= in_size_i do
              bi = bezier_blend(ki, mui, in_size_i)
              kj = 0
              while kj <= in_size_j do
                bj = bezier_blend(kj, muj, in_size_j)
                out[0] = out[0] + xyz_in[3 * (ki * (in_size_j + 1) + kj) + 0] * bi * bj
                out[1] = out[1] + xyz_in[3 * (ki * (in_size_j + 1) + kj) + 1] * bi * bj
                out[2] = out[2] + xyz_in[3 * (ki * (in_size_j + 1) + kj) + 2] * bi * bj
                kj = kj + 1
              end
              ki = ki + 1
            end
            out_idx = 3 * (i * out_size_j + j)
            xyz_outp[out_idx + 0] = out[0]
            xyz_outp[out_idx + 1] = out[1]
            xyz_outp[out_idx + 2] = out[2]
          end
          j = j + 1
        end
        i = i + 1
      end
      current = current + n_threads
      t = current
    end
  end

  defk bezier_surface(
    n_tasks, alpha, in_size_i,
    in_size_j, out_size_i, out_size_j,
    #xyz_l_in,
    xyz_in, xyz_outp
  ) do
    p[3]
    p[0] = n_tasks # p.n_tasks = n_tasks
    if alpha >= 0.0 && alpha <= 1.0 do
      p[1] = p[0] * alpha # p.cut = p.n_tasks * alpha
    end # no else!!!

    wg_in_J = (out_size_j - 1) / get_local_size(0) + 1;
    wg_in_I = (out_size_i - 1) / get_local_size(1) + 1;

    p[2] = p[1] + get_group_id(0) # p.current = p.cut + get_group_id(0)
    t = p[2]
    while p[2] < p[0] do # p.current < p.n_tasts
      my_s1 = t / wg_in_J
      my_s0 = t
      while my_s0 >= wg_in_J do
        my_s0 = my_s0 - wg_in_J # hack for %
      end

      row = my_s1 * get_local_size(1) + get_local_id(1)
      col = my_s0 * get_local_size(0) + get_local_id(0)
      bi = 0.0
      bj = 0.0
      mui = 1.0 * (row / (1.0 * (out_size_i - 1)))
      muj = 1.0 * (col / (1.0 * (out_size_j - 1)))

      if row < out_size_i && col < out_size_j do
        out[3]
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0

        ki = 0
        while ki <= in_size_i do
          bi = bezier_blend(ki, mui, in_size_i)

          kj = 0
          while kj <= in_size_j do
            bj = bezier_blend(kj, muj, in_size_j)

            out[0] = out[0] + (xyz_in[(ki * (in_size_j + 1) + kj) * 3 + 0] * bi * bj)
            out[1] = out[1] + (xyz_in[(ki * (in_size_j + 1) + kj) * 3 + 1] * bi * bj)
            out[2] = out[2] + (xyz_in[(ki * (in_size_j + 1) + kj) * 3 + 2] * bi * bj)
            kj = kj + 1
          end

          ki = ki + 1
        end

        xyz_outp[3 * (row * out_size_j + col) + 0] = out[0]
        xyz_outp[3 * (row * out_size_j + col) + 1] = out[1]
        xyz_outp[3 * (row * out_size_j + col) + 2] = out[2]
      end

      p[2] = p[2] + get_num_groups(0) # p.current = p.current + get_num_groups(0)
      t = p[2]
    end
  end

  def bs_gpu(
    n_work_items,
    n_work_groups,
    n_tasks,
    alpha,
    in_size_i,
    in_size_j,
    out_size_i,
    out_size_j,
    tensor
  ) do
    _in_len   = (in_size_i + 1) * (in_size_j + 1) * 3
    out_len  = out_size_i * out_size_j * 3

    ctx = Orchestra.gpu()

    in_gnx = Orchestra.new_gnx(ctx, tensor)
    out_gnx = Orchestra.new_gnx(ctx, Nx.broadcast(Nx.f32(0), {out_len}))

    start = System.monotonic_time()
    Orchestra.spawn(
      ctx,
      &BezierSurface.bezier_surface/8,

      {n_work_items * n_work_groups, n_work_items, 1},
      {n_work_items, n_work_items,1},

      [n_tasks, alpha,
       in_size_i, in_size_j, out_size_i, out_size_j,
       in_gnx, out_gnx]
    )
    stop = System.monotonic_time()
    IO.puts("gpu kernel took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms")

    Orchestra.get_gnx(ctx, out_gnx)
  end

  def bs_cpu(
    n_work_items,
    n_work_groups,
    n_tasks,
    alpha,
    in_size_i,
    in_size_j,
    out_size_i,
    out_size_j,
    tensor
  ) do
    _in_len   = (in_size_i + 1) * (in_size_j + 1) * 3
    out_len  = out_size_i * out_size_j * 3

    ctx = Orchestra.cpu()

    in_tensor = tensor
    out_tensor = Orchestra.tensor(ctx, {out_len}, type: {:f, 32}, fun: fn _ -> 0.0 end)

    start = System.monotonic_time()
    Orchestra.spawn(
      ctx,
      &BezierSurface.bezier_surface_cpu/9,
      {4, 1, 1},
      {1, 1, 1},
      [in_tensor, out_tensor,
       n_tasks, alpha, n_work_items,
       in_size_i, in_size_j,
       out_size_i, out_size_j,
      ]
    )
    stop = System.monotonic_time()
    IO.puts("cpu kernel took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms")

    out_tensor
  end
end

n_work_items = 16
n_work_groups = 32
n_tasks = 361
alpha = 0.000000
in_size_i = 3
in_size_j = 3
out_size_i = 300
out_size_j = 300
filepath = "benchmarks/cooperative/bs/control.txt"

tensor =
  filepath
  |> File.stream!()
  |> CSV.decode()
  |> Enum.to_list()
  |> Enum.map(fn {:ok, xs} -> xs end)
  |> Enum.map(fn xs -> Enum.map(xs, &String.to_float/1) end)
  |> List.flatten()
  |> Orchestra.tensor(type: {:f, 32})


start = System.monotonic_time()

cpu_task =
  Task.async(fn ->
    BezierSurface.bs_cpu(
      n_work_items,
      n_work_groups,
      n_tasks,
      alpha,
      in_size_i,
      in_size_j,
      out_size_i,
      out_size_j,
      tensor
    )
  end)

gpu_task =
  Task.async(fn ->
    BezierSurface.bs_gpu(
      n_work_items,
      n_work_groups,
      n_tasks,
      alpha,
      in_size_i,
      in_size_j,
      out_size_i,
      out_size_j,
      tensor
    )
  end)

out_cpu = Task.await(cpu_task, :infinity)
out_gpu = Task.await(gpu_task, :infinity)

merged = Nx.add(out_cpu, out_gpu)

stop = System.monotonic_time()

IO.puts("BS took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms")

expected_merged =
  "benchmarks/cooperative/bs/expected_merged.txt"
  |> File.stream!()
  |> CSV.decode()
  |> Enum.to_list()
  |> Enum.map(fn {:ok, xs} -> xs end)
  |> Enum.map(fn xs -> Enum.map(xs, &String.to_float/1) end)
  |> List.flatten()
  |> Nx.f32()

if Nx.all(Nx.less(Nx.abs(Nx.subtract(merged, expected_merged)), 0.000001)) == Nx.u8(1) do
  IO.puts "results are close enough"
else
  IO.puts "ERROR results are NOT close enough"
end

