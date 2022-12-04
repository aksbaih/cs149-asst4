#include "bfs.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstddef>
#include <unordered_set>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// #define VERBOSE

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) { list->count = 0; }

void vertex_set_init(vertex_set* list, int count) {
  list->max_vertices = count;
  list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

void bitmap_clear(bitmap* bm) { memset(bm->bits, 0, sizeof(bool) * bm->size); }

void bitmap_init(bitmap* bm, int count) {
  bm->bits = (bool*)malloc(sizeof(bool) * count);
  bm->size = count;
  bitmap_clear(bm);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* old_frontier,
    vertex_set* new_frontier,
    int* distances) {
  int i;
#pragma omp parallel for schedule(dynamic, 16)
  for (i = 0; i < old_frontier->count; i++) {
    int node = old_frontier->vertices[i];
    const Vertex* neighbor_begin = outgoing_begin(g, node);
    const Vertex* neighbor_end = outgoing_end(g, node);
    // attempt to add all neighbors to the new frontier
    const int neighbors_dis = distances[node] + 1;
    for (const Vertex* neighbor_ptr = neighbor_begin;
         neighbor_ptr < neighbor_end;
         neighbor_ptr++) {
      int neighbor = *neighbor_ptr;
      if (distances[neighbor] != NOT_VISITED_MARKER) continue;
      // the following operation should be atomic
      if (__sync_bool_compare_and_swap(
              &distances[neighbor], NOT_VISITED_MARKER, neighbors_dis)) {
        // the neighbor was not visited, but now it is.
        // now add the neighbor to the frontier.
        int index;
#pragma omp atomic capture
        index = new_frontier->count++;
        new_frontier->vertices[index] = neighbor;
      }
    }
  }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

int bottom_up_step(
    Graph g, bitmap* old_frontier_bm, bitmap* new_frontier_bm, int* distances) {
  int new_frontier_size = 0;
  // Iterate over unvesisited nodes
  int node;
#pragma omp parallel for schedule(dynamic, 256)
  for (node = 0; node < g->num_nodes; node++) {
    if (distances[node] != NOT_VISITED_MARKER) continue;
    // Check if node has an edge into the frontier.
    bool visit = false;
    const Vertex* parents_begin = incoming_begin(g, node);
    const Vertex* parents_end = incoming_end(g, node);
    const Vertex* parent_ptr;
    for (parent_ptr = parents_begin; parent_ptr < parents_end; parent_ptr++) {
      if (old_frontier_bm->bits[*parent_ptr]) {
        visit = true;
        break;
      }
    }
    if (visit) {
      // If it's visited now, add it to the new frontier.
      distances[node] = distances[*parent_ptr] + 1;
      new_frontier_bm->bits[node] = true;
#pragma omp atomic update
      new_frontier_size++;
    }
  }
  return new_frontier_size;
}

void bfs_bottom_up(Graph graph, solution* sol) {
  // CS149 students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.
  vertex_set list1;
  vertex_set list2;
  vertex_set list3;
  vertex_set list4;
  bitmap bm1;
  bitmap bm2;
  // vertex_set_init(&list1, graph->num_nodes);
  // vertex_set_init(&list2, graph->num_nodes);
  // vertex_set_init(&list3, graph->num_nodes);
  // vertex_set_init(&list4, graph->num_nodes);
  bitmap_init(&bm1, graph->num_nodes);
  bitmap_init(&bm2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;
  vertex_set* unvisited = &list3;
  vertex_set* new_unvisited = &list4;
  bitmap* frontier_bm = &bm1;
  bitmap* new_frontier_bm = &bm2;

  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  sol->distances[ROOT_NODE_ID] = 0;
  frontier_bm->bits[ROOT_NODE_ID] = true;

  int frontier_size = 1;
  while (frontier_size != 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
    int old_frontier_size = frontier_size;
#endif

    bitmap_clear(new_frontier_bm);

    frontier_size =
        bottom_up_step(graph, frontier_bm, new_frontier_bm, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf(
        "frontier=%-10d %.4f sec\n", old_frontier_size, end_time - start_time);
#endif

    // // swap pointers
    // vertex_set* tmp = frontier;
    // frontier = new_frontier;
    // new_frontier = tmp;
    // // swap unvisited
    // tmp = unvisited;
    // unvisited = new_unvisited;
    // new_unvisited = tmp;
    // swap bms
    bitmap* tmp_bm = frontier_bm;
    frontier_bm = new_frontier_bm;
    new_frontier_bm = tmp_bm;
  }
}

void bfs_hybrid(Graph graph, solution* sol) {
  // CS149 students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.
  vertex_set list1;
  vertex_set list2;
  bitmap bm1;
  bitmap bm2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);
  bitmap_init(&bm1, graph->num_nodes);
  bitmap_init(&bm2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;
  bitmap* frontier_bm = &bm1;
  bitmap* new_frontier_bm = &bm2;

  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  const int average_degree = graph->num_edges / graph->num_nodes;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  bool top_down = true;
  int frontier_size = 1;
  while (frontier_size != 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
    int old_frontier_size = frontier_size;
#endif

    // Advance the frontier by one step.
    if (top_down) {
      vertex_set_clear(new_frontier);
      top_down_step(graph, frontier, new_frontier, sol->distances);
      frontier_size = new_frontier->count;
      // swap pointers
      vertex_set* tmp = frontier;
      frontier = new_frontier;
      new_frontier = tmp;
    } else {  // bottom up
      bitmap_clear(new_frontier_bm);
      frontier_size =
          bottom_up_step(graph, frontier_bm, new_frontier_bm, sol->distances);
      // swap pointers
      bitmap* tmp_bm = frontier_bm;
      frontier_bm = new_frontier_bm;
      new_frontier_bm = tmp_bm;
    }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf(
        "frontier=%-10d %.4f sec\n", old_frontier_size, end_time - start_time);
#endif

    // Switch if necessary.
    if (top_down && (frontier_size > graph->num_nodes / average_degree)) {
#ifdef VERBOSE
      printf("Switching to bottom-up at frontier=%-10d.\n", frontier_size);
#endif
      top_down = false;
      bitmap_clear(frontier_bm);
      // Populate the frontier bitmap appropriately.
      int i;
      for (i = 0; i < frontier->count; i++) {
        frontier_bm->bits[frontier->vertices[i]] = true;
      }
    }
    // Possible support for switching back
    //     else if (
    //         !top_down
    //         && (frontier_size < graph->num_nodes / (average_degree + 1))) {
    // #ifdef VERBOSE
    //       printf("Switching to top-down at frontier=%-10d.\n",
    //       frontier_size);
    // #endif
    //       top_down = true;
    //       vertex_set_clear(frontier);
    //       // Populate the frontier bitmap appropriately.
    //       int i;
    //       for (i = 0; i < graph->num_nodes; i++) {
    //         if (frontier_bm->bits[i]) frontier->vertices[frontier->count++] =
    //         i;
    //       }
    //     }
  }
}
