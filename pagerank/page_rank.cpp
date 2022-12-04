#include "page_rank.h"

#include <omp.h>
#include <stdlib.h>

#include <cmath>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is
// num_nodes(g)) damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence) {
  int numNodes = num_nodes(g);
  const double dampingOffset = (1.0 - damping) / numNodes;
  double* scoresOdd = (double*)malloc(numNodes * sizeof(double));
  double* scoresEven = (double*)malloc(numNodes * sizeof(double));
  double* scoresArrays[2] = {scoresEven, scoresOdd};
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    scoresEven[i] = equal_prob;
  }

  bool converged = false;
  int iters = 1;
  while (!converged) {
    double* scoresOld = scoresArrays[(iters - 1) % 2];
    double* scoresNew = scoresArrays[iters % 2];

    double deadendOffset = 0;
    // Perf: this loop doesn't make a difference.
    // #pragma omp parallel for schedule(dynamic, 256) reduction(+ :
    // deadendOffset)
    for (int vi = 0; vi < numNodes; vi++) {
      if (outgoing_size(g, vi) == 0) {
        deadendOffset += damping * scoresOld[vi] / numNodes;
      }
    }

    double globalDiff = 0;
    int vi;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : globalDiff)
    for (vi = 0; vi < numNodes; vi++) {
      double newScore = 0;
      const Vertex* incomingStart = incoming_begin(g, vi);
      const Vertex* incomingEnd = incoming_end(g, vi);
      const Vertex* vj;
      // #pragma omp parallel for schedule(static, 80) reduction(+ : newScore)
      for (vj = incomingStart; vj != incomingEnd; vj++) {
        newScore += scoresOld[*vj] / outgoing_size(g, *vj);
      }

      newScore = damping * newScore + dampingOffset;

      newScore += deadendOffset;

      // Update diff
      scoresNew[vi] = newScore;
      globalDiff += abs(newScore - scoresOld[vi]);
    }

    converged = globalDiff < convergence;
    iters++;
  }
  double* scores = scoresArrays[(iters - 1) % 2];
  // Perf: doesn't make a difference
  memcpy(solution, scores, sizeof(double) * numNodes);
  free(scoresOdd);
  free(scoresEven);
}
