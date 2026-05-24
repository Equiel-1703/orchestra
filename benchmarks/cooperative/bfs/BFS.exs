require Orchestra

# Orchestra.set_debug_logs(true)

defmodule CsrReader do
  @header_regex ~r/^(?<n1>\d+)\s+(?<n2>\d+)\s+(?<n3>\d+)$/
  @pairs_regex ~r/^(?<n1>\d+)\s+(?<n2>\d+)$/

  defp parse_pair!(line) do
    case Regex.named_captures(@pairs_regex, line) do
      nil ->
        raise "regex error: expected a pair, found: '#{line}'"

      regex_map ->
        {String.to_integer(regex_map["n1"]), String.to_integer(regex_map["n2"])}
    end
  end

  # processing header
  defp process_line(line, %{reading_state: :header} = map) do
    case Regex.named_captures(@header_regex, line) do
      nil ->
        raise "regex error: expected the header, found: '#{line}'"

      regex_map ->
        # IO.inspect(regex_map, label: "Found header")

        n1 = String.to_integer(regex_map["n1"])
        n2 = String.to_integer(regex_map["n2"])
        n3 = String.to_integer(regex_map["n3"])

        Map.merge(map, %{
          total_nodes: n1,
          remaining_nodes: n1,
          total_edges: n2,
          remaining_edges: n2,
          start_node: n3,
          reading_state: :nodes
        })
    end
  end

  # processing nodes
  defp process_line(line, %{reading_state: :nodes} = map) do
    {n1, n2} = parse_pair!(line)

    # IO.puts("header already processed, reading nodes")

    nodes = [[n1, n2] | map.nodes]
    remaining_nodes = map.remaining_nodes - 1

    %{
      map
      | nodes: nodes,
        remaining_nodes: remaining_nodes,
        reading_state:
          if remaining_nodes <= 0 do
            :edges
          else
            :nodes
          end
    }
  end

  # processing edges
  defp process_line(line, %{reading_state: :edges} = map) do
    {n1, n2} = parse_pair!(line)

    # IO.puts("header already processed, reading edges")

    edges = [[n1, n2] | map.edges]
    remaining_edges = map.remaining_edges - 1

    if remaining_edges < 0 do
      raise "error: found more edges than defined in the header!"
    end

    %{
      map
      | edges: edges,
        remaining_edges: remaining_edges
    }
  end

  def read_and_process_file(file_path) do
    initial_map = %{
      reading_state: :header,
      nodes: [],
      edges: []
    }

    final_map =
      file_path
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.filter(fn l -> l != "" end)
      |> Enum.reduce(initial_map, &process_line/2)

    nodes_rev = Enum.reverse(final_map.nodes)
    edges_rev = Enum.reverse(final_map.edges)

    # Creating the map that will be returned
    %{
      start_node: final_map.start_node,
      total_nodes: final_map.total_nodes,
      total_edges: final_map.total_edges,
      nodes: Orchestra.tensor(nodes_rev, :s32),
      edges: Orchestra.tensor(edges_rev, :s32)
    }
  end
end

Orchestra.defmodule BFS do
  defk cpu_bfs_kernel(
         nodes,
         n_nodes,
         edges,
         frontier,
         frontier_size,
         new_frontier,
         atomic(next_slot),
         visited
       ) do
    tid = get_global_id(0)

    if tid >= frontier_size || tid >= n_nodes do
      return
    end

    # Get node to process in this thread
    node_idx = frontier[tid]
    # Mark it as visited
    visited[node_idx] = 1

    # -- Get node info --
    # Node edges index in edges array
    node_edges_idx = nodes[node_idx * 2 + 0]
    # Number of edges this node has
    node_num_edges = nodes[node_idx * 2 + 1]

    for i in range(node_edges_idx, node_edges_idx + node_num_edges) do
      # Getting child node
      dest_node_idx = edges[i * 2 + 0]

      # Check if this node was visited
      if visited[dest_node_idx] == 0 do
        # Mark as visited
        visited[dest_node_idx] = 1

        # Update empty slot and get the index we will use here
        idx = atomic_add_int(next_slot, 1)

        # Add this node to new frontier
        new_frontier[idx] = dest_node_idx
      end
    end
  end

  def bfs(
        %{
          total_nodes: total_nodes,
          start_node: start_node
        } = map,
        max_iterations \\ :infinity
      ) do
    IO.puts("============== Creating Tensors... ==============")

    start = System.monotonic_time()

    frontier_size = 1
    frontier = Orchestra.tensor({total_nodes}, :s32, fn _ -> start_node end)
    new_frontier = Orchestra.tensor({total_nodes}, :s32)
    visited = Orchestra.tensor({total_nodes}, :s32, fn _ -> 0 end)

    # This variable holds the next available index in the new_frontier
    next_idx = Orchestra.tensor([0], :s32)

    stop = System.monotonic_time()

    IO.puts(
      "Tensor creation took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms"
    )

    IO.puts("============== Starting Recursion ==============")

    bfs_recursion(map, frontier_size, frontier, new_frontier, next_idx, visited, max_iterations)
  end

  defp bfs_recursion(_map, 0, _frontier_a, _frontier_b, _next_idx, _visited, _max_iterations),
    do: :ok

  defp bfs_recursion(_map, _frontier_size, _frontier_a, _frontier_b, _next_idx, _visited, 0),
    do: :ok

  defp bfs_recursion(
         %{
           total_nodes: total_nodes,
           nodes: nodes_tensor,
           edges: edges_tensor
         } = map,
         frontier_size,
         frontier,
         new_frontier,
         next_idx,
         visited,
         max_iterations
       ) do
    Orchestra.with Orchestra.cpu() do
      Orchestra.spawn(
        &BFS.cpu_bfs_kernel/8,
        {frontier_size},
        {0},
        [
          nodes_tensor,
          total_nodes,
          edges_tensor,
          frontier,
          frontier_size,
          new_frontier,
          next_idx,
          visited
        ]
      )
    end

    # For the next iteration, the next free index will be reset
    new_next_idx = Orchestra.tensor([0], :s32)
    new_frontier_size = Nx.to_number(next_idx[0])

    IO.puts("=================================================")
    IO.inspect(frontier_size, label: "processed frontier size")
    IO.inspect(frontier, label: "frontier")
    IO.inspect(new_frontier, label: "new_frontier")
    IO.inspect(new_frontier_size, label: "size of new_frontier")
    IO.inspect(next_idx, label: "next_idx")
    IO.inspect(visited, label: "visited")

    remaining_iterations =
      cond do
        is_integer(max_iterations) -> max_iterations - 1
        true -> max_iterations
      end

    bfs_recursion(
      map,
      new_frontier_size,
      new_frontier,
      frontier,
      new_next_idx,
      visited,
      remaining_iterations
    )
  end
end

# Getting name of file to process. The user can specify
argv = System.argv()
argv_len = length(argv)

{file, it} =
  case argv_len do
    1 ->
      [f] = argv
      {f, :infinity}

    2 ->
      [f, i] = argv

      if is_integer(i) && i > 0 do
        {f, i}
      else
        {f, :infinity}
      end

    _ ->
      IO.puts("Usage: mix run #{Path.basename(__ENV__.file)} FILE_PATH [MAX_ITERATIONS]\n")

      IO.puts(
        "The MAX_ITERATIONS is an optional parameter that must be a positive number greater than 0. It specifies how many levels of the graph the algorithm is allowed to explore. If omitted, BFS will assume infinite iterations are allowed."
      )

      System.halt(0)
  end

IO.puts("--- Processing Input File '#{Path.basename(file)}' ---")

start = System.monotonic_time()
graph_map = CsrReader.read_and_process_file(file)
stop = System.monotonic_time()

IO.inspect(graph_map)

IO.puts(
  "Time taken to read input file: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms"
)

start = System.monotonic_time()
BFS.bfs(graph_map, it)
stop = System.monotonic_time()

IO.puts("BFS took: #{System.convert_time_unit(stop - start, :native, :millisecond)}ms")
