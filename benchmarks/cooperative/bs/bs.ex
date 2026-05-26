require OCLPolyHok

OCLPolyHok.defmodule BezierSurface do
  defd bezier_blend_gpu(k, mu, n) do
    nn = 0
    kn = 0
    nkn = 0
    blend = 1
    nn  = n
    kn  = k
    nkn = n - k

    while nn >= 1 do
      blend = blend * nn
      nn = nn - 1
      if kn > 1 do
        blend = blend / kn
        kn = kn - 1
      end
      if nkn > 1 do
        blend = blend / nkn
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

  defk bezier_surface(
    n_tasks, alpha, in_size_i,
    in_size_j, out_size_i, out_size_j,
    #xyz_l_in,
    xyz_in, xyz_outp
  ) do
    p[5]
    p[0] = n_tasks # p.n_tasks = n_tasks
    if alpha >= 0.0 && alpha <= 1.0 do
      p[1] = p[0] * alpha # p.cut = p.n_tasks * alpha
    end # no else!!!

    wg_in_J = (out_size_j - 1) / (get_local_size(0) + 1);
    wg_in_I = (out_size_i - 1) / (get_local_size(1) + 1);

    i = get_local_id(1) * get_local_size(0) + get_local_id(0)
    #while i < (in_size_i + 1) * (in_size_j + 1) do
    #  xyz_l_in[i] = xyz_in[i];
    #  i = i + get_local_size(0) * get_local_size(1)
    #end

    __syncthreads() # barrier(CLK_LOCAL_MEM_FENCE)

    p[2] = p[1] + get_group_id(0) # p.current = p.cut + get_group_id(0)
    t = p[2]
    while p[2] < p[0] do # p.current < p.n_tasts
      my_s1 = t / wg_in_J
      my_s0 = t
      while my_s0 > wg_in_J do
        my_s0 = my_s0 - wg_in_J # hack for %
      end

      #row = my_s1 * get_local_size(1) + get_local_id(1)
      #col = my_s0 * get_local_size(0) + get_local_id(0)
      row = my_s1 * get_local_size(1) + get_local_id(1)
      col = my_s0 * get_local_size(0) + get_local_id(0)
      bi = 0
      bj = 0
      mui = row / (out_size_i - 1)
      muj = row / (out_size_j - 1)

      if row < out_size_i && col < out_size_j do
        out[3]
        out[0] = 0
        out[1] = 0
        out[2] = 0

        ki = 0
        while ki <= in_size_i do
          bi = bezier_blend_gpu(ki, mui, in_size_i)

          kj = 0
          while kj <= in_size_i do
            bi = bezier_blend_gpu(kj, muj, in_size_j)

            out[0] = xyz_in[(ki * (in_size_j + 1) + kj) * 3 + 0] * bi * bj
            out[1] = xyz_in[(ki * (in_size_j + 1) + kj) * 3 + 1] * bi * bj
            out[2] = xyz_in[(ki * (in_size_j + 1) + kj) * 3 + 2] * bi * bj

            kj = kj + 1
          end

          ki = ki + 1
        end

        xyz_outp[(row * out_size_j + col) * 3 + 0] = out[0]
        xyz_outp[(row * out_size_j + col) * 3 + 1] = out[1]
        xyz_outp[(row * out_size_j + col) * 3 + 2] = out[2]
      end

      p[2] = p[2] + get_num_groups(0) # p.current = p.current + get_num_groups(0)
      t = p[2]
    end
  end

  def bs() do
    n_work_items = 16
    n_work_groups = 32

    n_tasks = 361
    alpha = 0.100000
    in_size_i = 3
    in_size_j = 3
    out_size_i = 300
    out_size_j = 300

    in_len   = (in_size_i + 1) * (in_size_j + 1) * 3
    out_len  = out_size_i * out_size_j * 3

    gpu_ctx = OCLPolyHok.gpu()

    tensor =
      "benchmarks/cooperative/bs/control.txt"
      |> File.stream!()
      |> CSV.decode()
      |> Enum.to_list()
      |> Enum.map(fn {:ok, xs} -> xs end)
      |> Enum.map(fn xs -> Enum.map(xs, &String.to_float/1) end)
      |> Nx.tensor()
      |> Nx.reshape({16 * 3})

    IO.write("n_work_items: ")
    IO.inspect(n_work_items)
    IO.write("n_work_groups: ")
    IO.inspect(n_work_groups)
    IO.write("n_tasks: ")
    IO.inspect(n_tasks)
    IO.write("alpha: ")
    IO.inspect(alpha)
    IO.write("in_size_i: ")
    IO.inspect(in_size_i)
    IO.write("in_size_j: ")
    IO.inspect(in_size_j)
    IO.write("out_size_i: ")
    IO.inspect(out_size_i)
    IO.write("out_size_j: ")
    IO.inspect(out_size_j)

    in_size = in_len * 4
    out_size = out_len * 4
    IO.write("in_size: ")
    IO.inspect(in_size)
    IO.write("out_size: ")
    IO.inspect(out_size)

    IO.inspect(Nx.reshape(tensor, {16, 3}))

    in_gnx = OCLPolyHok.new_gnx(OCLPolyHok.gpu(), tensor)
    out_gnx = OCLPolyHok.new_gnx(gpu_ctx, {out_len}, {:f, 32})

    OCLPolyHok.spawn(
      gpu_ctx,
      &BezierSurface.bezier_surface/8,

      {n_work_items * n_work_groups, n_work_items, 1},
      {n_work_items, n_work_items,1},

      [n_tasks, alpha,
       in_size_i, in_size_j, out_size_i, out_size_j,
       in_gnx, out_gnx]
    )

    IO.inspect(Nx.to_binary(OCLPolyHok.get_gnx(gpu_ctx, out_gnx)), limit: 128)
  end
end

BezierSurface.bs()
