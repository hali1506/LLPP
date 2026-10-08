//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <omp.h>

#include <iterator>
#include <map>
#include <set>
#include <vector>

#include "cuda_ped_agents.h"
#include "ped_agents.h"

namespace Ped {

class Model {
 public:
  // Sets everything up
  void setup(std::vector<AgentsData> agentsInScenario,
             std::vector<Twaypoint*> destinationsInScenario,
             IMPLEMENTATION implementation,
             IMPLEMENTATION_MOVE moveImplementation);

  // Coordinates a time step in the scenario: move all agents by one step (if
  // applicable).
  void tick();

#ifndef NOCUDA
  void cudaTick();
#endif
  void simdTick();

  void cppthreadsTick(int nr_agents);

  void ompTick(int nr_agents);

  void seqTick(int nr_agents);

  // Returns the agents of this scenario
  const Tagents* getTagents() const { return tagents; };

  // Returns the number of agents
  // int getNumAgents() const { return tagents ? tagents->getNumAgents() : 0; };

  // Adds an agent to the tree structure
  void placeAgent(int agentIndex);

  // Cleans up the tree and restructures it. Worth calling every now and then.
  void cleanup();
  ~Model();

  // Returns the heatmap visualizing the density of agents
  int const* const* getHeatmap() const { return blurred_heatmap; };
  int getHeatmapSize() const;

 private:
  // Denotes which implementation (sequential, parallel implementations..)
  // should be used for calculating the desired positions of
  // agents (Assignment 1)
  IMPLEMENTATION implementation;
  IMPLEMENTATION_MOVE moveImplementation;

  // The agents in this scenario
  Tagents* tagents = nullptr;

  // The waypoints in this scenario
  std::vector<Twaypoint*> destinations;

  // Moves an agent towards its next position
  void move(int agentIndex);

  void processTile(int x, int y, int tx, int ty, int width, int height,
                   int numVBorders, int borderWidth,
                   std::vector<std::vector<int>>& tileAgents,
                   omp_lock_t* borderLocks, Ped::Tagents* tagents);

  void moveAllOMP();

  // CUDA
#ifndef NOCUDA
  CUDAPed* cudaPed = nullptr;
#endif

  unsigned int activeThreads = 0;

  int tilesX = 0;
  int tilesY = 0;
  int numVBorders = 0;
  int numHBorders = 0;
  int numBorderMutexes = 0;
  omp_lock_t* borderLocks = nullptr;

  ////////////
  /// Everything below here won't be relevant until Assignment 3
  ///////////////////////////////////////////////

  // Returns the set of neighboring agents for the specified position
  std::set<int> getNeighbors(int x, int y, int dist) const;

  ////////////
  /// Everything below here won't be relevant until Assignment 4
  ///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE* CELLSIZE

  // The heatmap representing the density of agents
  int** heatmap;

  // The scaled heatmap that fits to the view
  int** scaled_heatmap;

  // The final heatmap: blurred and scaled to fit the view
  int** blurred_heatmap;

  void setupHeatmapSeq();
  void updateHeatmapSeq();
};
}  // namespace Ped
#endif
